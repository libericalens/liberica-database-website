import os
import sys
import math
import io
from datetime import datetime, timezone, timedelta

import cv2
import pandas as pd
import numpy as np
import pytz
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from skimage import measure
from keras.utils import normalize

from UNet_Model import unet_model

# Set encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf8')

# Configure matplotlib to work without GUI
matplotlib.use('Agg')


# --------------------------
# Configuration and Setup
# --------------------------

class Config:
    """Application configuration"""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///liberica_bean_metadata_dummy2.db'
    TIMEZONE = 'Asia/Manila'
    UPLOAD_FOLDER = 'uploads'
    WATERSHED_FOLDER = 'for_watershed'
    MODEL_WEIGHTS = 'unet_model/coffee_bean_test-20.hdf5'
    IMAGE_SIZE = (256, 256)
    RECORDS_PER_PAGE = 100


# --------------------------
# Database Models
# --------------------------

class BaseModel:
    """Base model with common functionality"""
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: get_current_timestamp())


class LibericaBeanMetadata(db.Model, BaseModel):
    """Database model for Liberica Bean Metadata"""
    __tablename__ = 'liberica_bean_metadata'
    
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String)
    image_id = db.Column(db.String)
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
    class_label = db.Column(db.String)


# --------------------------
# Utility Functions
# --------------------------

def get_current_timestamp():
    """Get the current timestamp with timezone"""
    local_tz = pytz.timezone(Config.TIMEZONE)
    return datetime.now(timezone.utc).astimezone(local_tz).replace(microsecond=0)


def resize_image(filepath, output_path, size=Config.IMAGE_SIZE):
    """Resize image to the specified size"""
    img = Image.open(filepath)
    img = img.resize(size, Image.Resampling.LANCZOS)
    img.save(output_path)


def ensure_directory_exists(directory):
    """Ensure a directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# --------------------------
# Core Business Logic
# --------------------------

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
        model.load_weights(Config.MODEL_WEIGHTS)
        return model

    def load_and_process_image(self, filepath):
        """Load and process image for analysis"""
        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_norm = np.expand_dims(normalize(np.array(img), axis=1), 2)
        img_norm = img_norm[:, :, 0][:, :, None]
        return np.expand_dims(img_norm, 0)

    def segment_image(self, img):
        """Segment image using the model"""
        return (self.model.predict(img)[0, :, :, 0] > 0.9).astype(np.uint8)

    def save_segmented_image(self, segmented, output_filename):
        """Save segmented image to file"""
        plt.imsave(output_filename, segmented, cmap='gray')

    def apply_watershed_algorithm(self, img):
        """Apply watershed algorithm for bean separation"""
        img_grey = img[:, :, 0]
        ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret2, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
        sure_fg = np.array(sure_fg, dtype=np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 10
        markers[unknown == 255] = 0
        cv2.watershed(img, markers)
        return markers

    def extract_properties(self, markers, img):
        """Extract bean properties from markers"""
        img_grey = img[:, :, 0]
        return measure.regionprops_table(
            markers, 
            intensity_image=img_grey,
            properties=[
                'area', 'perimeter', 'equivalent_diameter', 'extent', 
                'mean_intensity', 'solidity', 'convex_area', 
                'axis_major_length', 'axis_minor_length', 'eccentricity'
            ]
        )

    def create_bean_dataframe(self, props, class_label, filepath):
        """Create a pandas DataFrame from the extracted properties"""
        df = pd.DataFrame(props)
        df = df[df.mean_intensity > 100]  # Filter out background/noise
        df['class_label'] = class_label
        image_id = os.path.basename(filepath)
        
        columns = [
            'filepath', 'image_id', 'area', 'perimeter', 'equivalent_diameter',
            'extent', 'mean_intensity', 'solidity', 'convex_area',
            'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label'
        ]
        
        df.insert(0, 'filepath', filepath)
        df.insert(1, 'image_id', image_id)
        return df[columns]

    def save_to_database(self, df):
        """Save the DataFrame to the database"""
        records = df.to_dict('records')
        for record in records:
            metadata = LibericaBeanMetadata(**record)
            db.session.add(metadata)
        db.session.commit()


# --------------------------
# Flask Application Setup
# --------------------------

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)


# --------------------------
# View Helpers
# --------------------------

class DashboardHelper:
    """Helper class for dashboard view logic"""
    @staticmethod
    def load_recent_data(months=5):
        """Load recent data from the database"""
        cutoff_date = datetime.now() - timedelta(days=30*months)
        query = LibericaBeanMetadata.query.filter(LibericaBeanMetadata.created_at >= cutoff_date)
        return pd.DataFrame(
            [(d.created_at, d.area, d.perimeter, d.equivalent_diameter, d.extent, 
              d.axis_major_length, d.axis_minor_length, d.eccentricity, d.class_label) 
             for d in query.all()],
            columns=['created_at', 'area', 'perimeter', 'equivalent_diameter', 'extent', 
                    'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label']
        )

    @staticmethod
    def calculate_monthly_averages(data):
        """Calculate monthly averages from the data"""
        data['created_at'] = pd.to_datetime(data['created_at'])
        avg_data = data.groupby([pd.Grouper(key='created_at', freq='M'), 'class_label']).mean().reset_index()
        avg_data['created_at'] = avg_data['created_at'].dt.strftime('%Y-%m')
        return avg_data

    @staticmethod
    def create_graph(avg_data, selected_feature, class_label):
        """Create Plotly graph for the dashboard"""
        fig = go.Figure()
        
        if not selected_feature or not class_label:
            return fig.to_html(full_html=False)
            
        filtered_data = avg_data[avg_data['class_label'] == class_label]
        
        if selected_feature == 'Area':
            for feature in avg_data.columns[1:]:
                fig.add_trace(go.Bar(
                    x=filtered_data['created_at'], 
                    y=filtered_data[feature], 
                    name=feature.replace('_', ' ').capitalize()
                ))
        else:
            fig.add_trace(go.Bar(
                x=filtered_data['created_at'], 
                y=filtered_data[selected_feature], 
                name=selected_feature.replace('_', ' ').capitalize(), 
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


class RecordsHelper:
    """Helper class for records view logic"""
    @staticmethod
    def get_paginated_records(page=1, sort_by='id', sort_order='asc', class_label=None):
        """Get paginated and sorted records from the database"""
        query = LibericaBeanMetadata.query
        
        if class_label:
            query = query.filter_by(class_label=class_label)
            
        if sort_order == 'asc':
            query = query.order_by(getattr(LibericaBeanMetadata, sort_by))
        else:
            query = query.order_by(getattr(LibericaBeanMetadata, sort_by).desc())
            
        total_records = query.count()
        total_pages = math.ceil(total_records / Config.RECORDS_PER_PAGE)
        
        paginated_data = query.paginate(
            page=page, 
            per_page=Config.RECORDS_PER_PAGE,
            error_out=False
        )
        
        return {
            'data': paginated_data.items,
            'page': page,
            'total_pages': total_pages,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'class_label': class_label,
            'class_labels': db.session.query(LibericaBeanMetadata.class_label).distinct().all()
        }


# --------------------------
# Flask Routes
# --------------------------

@app.route('/')
def homepage():
    """Render the homepage"""
    return render_template("homepage.html")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Render the dashboard with bean analysis data"""
    data = DashboardHelper.load_recent_data()
    avg_data = DashboardHelper.calculate_monthly_averages(data)
    
    # Handle feature selection
    selected_feature = request.form.get('feature')
    class_label = request.form.get('class_label')
    
    graph_div = DashboardHelper.create_graph(avg_data, selected_feature, class_label)
    class_labels = avg_data['class_label'].unique().tolist()
    features = [col for col in avg_data.columns[1:] if col != 'class_label']
    
    return render_template(
        "dashboard.html",
        graph_div=graph_div,
        features=features,
        title=f'Liberica Bean Metadata Dashboard - {selected_feature} - {class_label}'.title(),
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
    page = int(request.args.get('page', 1))
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'asc')
    class_label = request.args.get('class_label')
    
    context = RecordsHelper.get_paginated_records(page, sort_by, sort_order, class_label)
    return render_template('records.html', **context)


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    """Handle coffee bean scanning and analysis"""
    if request.method == 'GET':
        return render_template('scan_coffee_bean.html')
        
    # Handle POST request
    file = request.files['file']
    class_label = request.form['class_label']
    
    # Ensure upload directories exist
    ensure_directory_exists(Config.UPLOAD_FOLDER)
    ensure_directory_exists(Config.WATERSHED_FOLDER)
    
    # Process uploaded file
    filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Resize and analyze image
    output_path = os.path.join(Config.UPLOAD_FOLDER, f'{file.filename}.png')
    resize_image(filepath, output_path)
    
    analyzer = CoffeeBeanAnalyzer()
    img = analyzer.load_and_process_image(output_path)
    segmented = analyzer.segment_image(img)
    
    # Save and process segmented image
    output_filename = os.path.join(Config.WATERSHED_FOLDER, file.filename)
    analyzer.save_segmented_image(segmented, output_filename)
    
    # Extract properties and save to database
    img = cv2.imread(output_filename)
    markers = analyzer.apply_watershed_algorithm(img)
    props = analyzer.extract_properties(markers, img)
    df = analyzer.create_bean_dataframe(props, class_label, output_path)
    analyzer.save_to_database(df)
    
    return render_template('results.html', df=df)


# --------------------------
# Application Entry Point
# --------------------------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
