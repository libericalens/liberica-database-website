import os
import sys
import math
import io
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import pandas as pd
import pytz
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from keras.utils import normalize
from matplotlib import pyplot as plt
from PIL import Image
from skimage import measure
import plotly.graph_objects as go
from UNet_Model import unet_model

# Configure matplotlib to work without GUI
plt.switch_backend('Agg')

# Set encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf8')

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///liberica_bean_metadata_dummy2.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['WATERSHED_FOLDER'] = 'for_watershed'

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['WATERSHED_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Constants
TIMEZONE = pytz.timezone('Asia/Manila')
DEFAULT_IMAGE_SIZE = (256, 256)
MODEL_WEIGHTS_PATH = 'unet_model/coffee_bean_test-20.hdf5'
PAGE_SIZE = 100


def get_current_timestamp():
    """Get the current timestamp with timezone"""
    return datetime.now(timezone.utc).astimezone(TIMEZONE).replace(microsecond=0)


class LibericaBeanMetadata(db.Model):
    """Database model for Liberica Bean Metadata"""
    __tablename__ = 'liberica_bean_metadata'
    
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(255))
    image_id = db.Column(db.String(255))
    area = db.Column(db.Float)
    perimeter = db.Column(db.Float)
    equivalent_diameter = db.Column(db.Float)
    extent = db.Column(db.Float)
    mean_intensity = db.Column(db.Float)
    solidity = db.Column(db.Float)
    convex_area = db.Column(db.Float)
    axis_major_length = db.Column(db.Float)
    axis_minor_length = db.Column(db.Float)
    eccentricity = db.Column(db.Float)
    class_label = db.Column(db.String(50))
    created_at = db.Column(db.DateTime(timezone=True), default=get_current_timestamp)


class CoffeeBeanAnalyzer:
    """Class for analyzing coffee beans"""
    
    def __init__(self, img_height=256, img_width=256, img_channels=1):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.IMG_CHANNELS = img_channels
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and load the U-Net model"""
        model = unet_model(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)
        model.load_weights(MODEL_WEIGHTS_PATH)
        return model
    
    def load_and_process_image(self, filepath):
        """Load and preprocess image for model input"""
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image from {filepath}")
            
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
            img_norm = np.expand_dims(normalize(np.array(img), axis=1), 2)
            img_norm = img_norm[:, :, 0][:, :, None]
            return np.expand_dims(img_norm, 0)
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def segment_image(self, img):
        """Segment image using the model"""
        return (self.model.predict(img, verbose=0)[0, :, :, 0] > 0.9).astype(np.uint8)
    
    @staticmethod
    def save_segmented_image(segmented, output_filename):
        """Save segmented image to file"""
        plt.imsave(output_filename, segmented, cmap='gray')
    
    @staticmethod
    def apply_watershed_algorithm(img):
        """Apply watershed algorithm to segment individual beans"""
        if len(img.shape) == 3:
            img_grey = img[:, :, 0]
        else:
            img_grey = img
            
        _, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 10
        markers[unknown == 255] = 0
        
        if len(img.shape) == 3:
            cv2.watershed(img, markers)
        return markers
    
    @staticmethod
    def extract_properties(markers, img):
        """Extract region properties from segmented image"""
        if len(img.shape) == 3:
            img_grey = img[:, :, 0]
        else:
            img_grey = img
            
        return measure.regionprops_table(
            markers, 
            intensity_image=img_grey,
            properties=[
                'area', 'perimeter', 'equivalent_diameter', 'extent', 
                'mean_intensity', 'solidity', 'convex_area', 
                'axis_major_length', 'axis_minor_length', 'eccentricity'
            ]
        )
    
    @staticmethod
    def create_dataframe(props, class_label, filepath):
        """Create a pandas DataFrame from the extracted properties"""
        df = pd.DataFrame(props)
        df = df[df['mean_intensity'] > 100]  # Filter out background/noise
        
        image_id = os.path.basename(filepath)
        df['class_label'] = class_label
        df['filepath'] = filepath
        df['image_id'] = image_id
        
        # Reorder columns
        columns = [
            'filepath', 'image_id', 'area', 'perimeter', 'equivalent_diameter', 
            'extent', 'mean_intensity', 'solidity', 'convex_area', 
            'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label'
        ]
        return df[columns]
    
    @staticmethod
    def save_to_database(df):
        """Save the DataFrame to the database"""
        try:
            records = df.to_dict('records')
            for record in records:
                metadata = LibericaBeanMetadata(**record)
                db.session.add(metadata)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Database error: {str(e)}")


def resize_image(filepath, output_path, size=DEFAULT_IMAGE_SIZE):
    """Resize image to the specified size"""
    try:
        with Image.open(filepath) as img:
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(output_path)
    except Exception as e:
        raise ValueError(f"Error resizing image: {str(e)}")


def load_recent_data(days=150):
    """Load recent data from the database"""
    time_threshold = datetime.now() - timedelta(days=days)
    query = LibericaBeanMetadata.query.filter(LibericaBeanMetadata.created_at >= time_threshold)
    data = query.all()
    
    return pd.DataFrame(
        [(d.created_at, d.area, d.perimeter, d.equivalent_diameter, d.extent, 
          d.axis_major_length, d.axis_minor_length, d.eccentricity, d.class_label) 
         for d in data],
        columns=['created_at', 'area', 'perimeter', 'equivalent_diameter', 'extent', 
                'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label']
    )


def calculate_monthly_averages(data):
    """Calculate monthly averages from the data"""
    data['created_at'] = pd.to_datetime(data['created_at'])
    avg_data = data.groupby([pd.Grouper(key='created_at', freq='M'), 'class_label']).mean().reset_index()
    avg_data['created_at'] = avg_data['created_at'].dt.strftime('%Y-%m')
    return avg_data


def create_plot(avg_data, selected_feature, class_label):
    """Create Plotly visualization"""
    fig = go.Figure()
    
    if not selected_feature or not class_label:
        return fig.to_html(full_html=False)
    
    filtered_data = avg_data[avg_data['class_label'] == class_label]
    if filtered_data.empty:
        return fig.to_html(full_html=False)
    
    if selected_feature.lower() == 'area':
        for feature in filtered_data.columns[1:]:
            if feature != 'class_label':
                fig.add_trace(go.Bar(
                    x=filtered_data['created_at'], 
                    y=filtered_data[feature], 
                    name=feature.replace('_', ' ').title()
                ))
    else:
        fig.add_trace(go.Bar(
            x=filtered_data['created_at'], 
            y=filtered_data[selected_feature], 
            name=selected_feature.replace('_', ' ').title(), 
            marker_color='#591f0b'
        ))
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array', 
            tickvals=filtered_data['created_at'], 
            ticktext=filtered_data['created_at']
        ), 
        yaxis=dict(visible=False)
    )
    
    return fig.to_html(full_html=False)


@app.route('/')
def homepage():
    """Render the homepage"""
    return render_template("homepage.html")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Render the dashboard with data visualization"""
    data = load_recent_data()
    avg_data = calculate_monthly_averages(data)
    
    # Handle feature selection
    selected_feature = request.form.get('feature', 'area')
    class_label = request.form.get('class_label', avg_data['class_label'].iloc[0] if not avg_data.empty else '')
    
    graph_div = create_plot(avg_data, selected_feature, class_label)
    class_labels = avg_data['class_label'].unique().tolist()
    
    # Available features (excluding date and class label)
    features = [col for col in avg_data.columns if col not in ['created_at', 'class_label']]
    
    return render_template(
        "dashboard.html", 
        graph_div=graph_div, 
        features=features, 
        title=f'Liberica Bean Metadata - {selected_feature} - {class_label}'.title(), 
        selected_feature=selected_feature, 
        class_label=class_label, 
        class_labels=class_labels
    )


@app.route('/objectives')
def objectives():
    """Render the objectives page"""
    return render_template('objectives.html')


@app.route('/records')
def records():
    """Render the records page with pagination and sorting"""
    try:
        # Get query parameters with defaults
        page = max(1, int(request.args.get('page', 1)))
        sort_by = request.args.get('sort_by', 'id')
        sort_order = request.args.get('sort_order', 'asc')
        class_label = request.args.get('class_label')

        # Base query
        query = LibericaBeanMetadata.query

        # Apply class label filter if specified
        if class_label:
            query = query.filter(LibericaBeanMetadata.class_label == class_label)

        # Apply sorting
        if hasattr(LibericaBeanMetadata, sort_by):
            if sort_order == 'desc':
                query = query.order_by(getattr(LibericaBeanMetadata, sort_by).desc())
            else:
                query = query.order_by(getattr(LibericaBeanMetadata, sort_by))

        # Get unique class labels for filter dropdown
        class_labels = db.session.query(LibericaBeanMetadata.class_label).distinct().all()
        class_labels = [label[0] for label in class_labels]

        # Pagination
        paginated_data = query.paginate(page=page, per_page=PAGE_SIZE, error_out=False)
        
        # Ensure we have valid pagination data
        if not paginated_data.items:
            return render_template('records.html', 
                                 data=[], 
                                 pagination=None,
                                 sort_by=sort_by, 
                                 sort_order=sort_order, 
                                 class_labels=class_labels, 
                                 class_label=class_label)

        return render_template(
            'records.html', 
            data=paginated_data.items, 
            pagination=paginated_data,
            sort_by=sort_by, 
            sort_order=sort_order, 
            class_labels=class_labels, 
            class_label=class_label,
            page=page  # Explicitly pass page variable
        )
    except Exception as e:
        app.logger.error(f"Error in records route: {str(e)}")
        return render_template('error.html', error="An error occurred while loading records"), 500

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    """Handle coffee bean scanning and analysis"""
    if request.method == 'POST':
        try:
            # Validate file upload
            if 'file' not in request.files:
                return render_template('scan_coffee_bean.html', error='No file uploaded')
            
            file = request.files['file']
            if not file.filename:
                return render_template('scan_coffee_bean.html', error='No file selected')
            
            # Validate class label
            class_label = request.form.get('class_label')
            if not class_label:
                return render_template('scan_coffee_bean.html', error='Class label is required')
            
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Resize image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'resized_{file.filename}')
            resize_image(filepath, output_path)
            
            # Analyze the image
            analyzer = CoffeeBeanAnalyzer()
            img = analyzer.load_and_process_image(output_path)
            segmented = analyzer.segment_image(img)
            
            # Save segmented image
            output_filename = os.path.join(app.config['WATERSHED_FOLDER'], f'segmented_{file.filename}')
            analyzer.save_segmented_image(segmented, output_filename)
            
            # Apply watershed algorithm
            img = cv2.imread(output_filename)
            markers = analyzer.apply_watershed_algorithm(img)
            props = analyzer.extract_properties(markers, img)
            
            # Create and save results
            df = analyzer.create_dataframe(props, class_label, output_path)
            analyzer.save_to_database(df)
            
            return render_template('results.html', df=df)
        
        except Exception as e:
            return render_template('scan_coffee_bean.html', error=str(e))
    
    return render_template('scan_coffee_bean.html')


def create_app():
    """Application factory function"""
    with app.app_context():
        db.create_all()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
