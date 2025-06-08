import os
import sys
import math
import io
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

import cv2
import pandas as pd
import numpy as np
import pytz
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from skimage import measure
from keras.utils import normalize
from concurrent.futures import ThreadPoolExecutor
import traceback

from UNet_Model import unet_model

# Configure environment
matplotlib.use('Agg')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf8')

# --------------------------
# Configuration
# --------------------------

class Config:
    """Optimized application configuration"""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///liberica_bean_metadata_optimized.db'
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_pre_ping': True,
        'pool_recycle': 3600
    }
    TIMEZONE = 'Asia/Manila'
    UPLOAD_FOLDER = 'uploads'
    WATERSHED_FOLDER = 'for_watershed'
    MODEL_WEIGHTS = 'unet_model/coffee_bean_test-20.hdf5'
    IMAGE_SIZE = (256, 256)
    RECORDS_PER_PAGE = 100
    MAX_WORKERS = 4  # For parallel processing
    WATERSHED_ITERATIONS = 10
    INTENSITY_THRESHOLD = 100  # For noise filtering
    SECRET_KEY = 'your-secret-key-here'  # Required for flash messages

# --------------------------
# Application Setup
# --------------------------

app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = Config.SQLALCHEMY_ENGINE_OPTIONS
db = SQLAlchemy(app)

# Thread pool for parallel tasks
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

# --------------------------
# Database Models
# --------------------------

class LibericaBeanMetadata(db.Model):
    """Optimized database model with indexes"""
    __tablename__ = 'liberica_bean_metadata_opt'
    
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(255), index=True)
    image_id = db.Column(db.String(100), index=True)
    area = db.Column(db.Float)
    perimeter = db.Column(db.Float)
    equivalent_diameter = db.Column(db.Float)
    extent = db.Column(db.Float)
    mean_intensity = db.Column(db.Float, index=True)
    solidity = db.Column(db.Float)
    convex_area = db.Column(db.Float)
    axis_major_length = db.Column(db.Float)
    axis_minor_length = db.Column(db.Float)
    eccentricity = db.Column(db.Float)
    class_label = db.Column(db.String(50), index=True)
    created_at = db.Column(db.DateTime(timezone=True), 
                         default=lambda: datetime.now(pytz.timezone(Config.TIMEZONE)),
                         index=True)

# --------------------------
# Core Analysis Engine
# --------------------------

class CoffeeBeanAnalyzer:
    """Optimized coffee bean analyzer with caching and batch processing"""
    _model_instance = None
    
    def __init__(self):
        self._initialize_directories()
        
    @classmethod
    def get_model(cls):
        """Singleton model instance with lazy loading"""
        if cls._model_instance is None:
            try:
                cls._model_instance = unet_model(256, 256, 1)
                cls._model_instance.load_weights(Config.MODEL_WEIGHTS)
                app.logger.info("Model loaded successfully")
            except Exception as e:
                app.logger.error(f"Failed to load model: {str(e)}")
                raise
        return cls._model_instance
    
    def _initialize_directories(self):
        """Ensure required directories exist"""
        try:
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(Config.WATERSHED_FOLDER, exist_ok=True)
            os.makedirs(os.path.dirname(Config.MODEL_WEIGHTS), exist_ok=True)
        except Exception as e:
            app.logger.error(f"Failed to create directories: {str(e)}")
            raise
    
    @staticmethod
    def _resize_image(filepath, output_path):
        """Optimized image resizing with memory efficiency"""
        try:
            with Image.open(filepath) as img:
                img = img.resize(Config.IMAGE_SIZE, Image.Resampling.LANCZOS)
                img.save(output_path, optimize=True, quality=85)
        except Exception as e:
            app.logger.error(f"Failed to resize image: {str(e)}")
            raise
    
    def process_uploaded_file(self, file, class_label):
        """Optimized file processing pipeline"""
        try:
            # Generate unique filename to prevent collisions
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            original_filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, original_filename)
            
            # Save original file
            file.save(filepath)
            
            # Process in optimized pipeline
            output_path = os.path.join(Config.UPLOAD_FOLDER, f"processed_{original_filename}.png")
            self._resize_image(filepath, output_path)
            
            # Process image asynchronously
            future = executor.submit(self._analyze_image, output_path, class_label)
            return future
        except Exception as e:
            app.logger.error(f"Failed to process uploaded file: {str(e)}")
            raise
    
    def _analyze_image(self, filepath, class_label):
        """Core analysis pipeline optimized for performance"""
        try:
            # Load and process image
            img_input = self._load_and_process_image(filepath)
            
            # Segment image
            segmented = self._segment_image(img_input)
            
            # Save segmented image
            output_filename = os.path.join(Config.WATERSHED_FOLDER, os.path.basename(filepath))
            self._save_segmented_image(segmented, output_filename)
            
            # Watershed processing
            markers = self._apply_watershed(output_filename)
            
            # Extract and save properties
            return self._extract_and_save_properties(markers, output_filename, class_label, filepath)
        except Exception as e:
            app.logger.error(f"Error processing image {filepath}: {traceback.format_exc()}")
            raise
    
    def _load_and_process_image(self, filepath):
        """Optimized image loading with memory mapping"""
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image from {filepath}")
            img = cv2.resize(img, (256, 256))
            img_norm = normalize(np.array(img, dtype=np.float32), axis=1)
            return np.expand_dims(img_norm[:, :, np.newaxis], 0)
        except Exception as e:
            app.logger.error(f"Failed to load and process image: {str(e)}")
            raise
    
    def _segment_image(self, img):
        """Batch-optimized segmentation"""
        try:
            return (self.get_model().predict(img, batch_size=1)[0, :, :, 0] > 0.9).astype(np.uint8)
        except Exception as e:
            app.logger.error(f"Failed to segment image: {str(e)}")
            raise
    
    def _save_segmented_image(self, segmented, output_filename):
        """Optimized image saving"""
        try:
            plt.imsave(output_filename, segmented, cmap='gray', format='png', dpi=100)
        except Exception as e:
            app.logger.error(f"Failed to save segmented image: {str(e)}")
            raise
    
    def _apply_watershed(self, image_path):
        """Optimized watershed algorithm with pre-allocated arrays"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            img_grey = img[:, :, 0]
            
            # Thresholding
            _, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=Config.WATERSHED_ITERATIONS)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            
            # Foreground extraction
            _, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            
            # Marker creation
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers += 10
            markers[unknown == 255] = 0
            
            # Watershed
            cv2.watershed(img, markers)
            return markers
        except Exception as e:
            app.logger.error(f"Failed to apply watershed: {str(e)}")
            raise
    
    def _extract_and_save_properties(self, markers, image_path, class_label, original_path):
        """Batch property extraction and optimized database insertion"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            img_grey = img[:, :, 0]
            
            props = measure.regionprops_table(
                markers,
                intensity_image=img_grey,
                properties=[
                    'area', 'perimeter', 'equivalent_diameter', 'extent',
                    'mean_intensity', 'solidity', 'convex_area',
                    'axis_major_length', 'axis_minor_length', 'eccentricity'
                ]
            )
            
            df = pd.DataFrame(props)
            df = df[df.mean_intensity > Config.INTENSITY_THRESHOLD]
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Prepare optimized data structure
            df['class_label'] = class_label
            df['filepath'] = original_path
            df['image_id'] = os.path.basename(original_path)
            df['created_at'] = datetime.now(pytz.timezone(Config.TIMEZONE))
            
            # Batch insert to database
            self._batch_insert_to_db(df)
            return df
        except Exception as e:
            app.logger.error(f"Failed to extract and save properties: {str(e)}")
            raise
    
    def _batch_insert_to_db(self, df):
        """Optimized bulk database insertion"""
        try:
            records = df.to_dict('records')
            if not records:
                return
            
            # Use bulk_insert_mappings for better performance
            db.session.bulk_insert_mappings(LibericaBeanMetadata, records)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Failed to insert records to database: {str(e)}")
            raise

# --------------------------
# View Helpers (Optimized)
# --------------------------

class DashboardHelper:
    """Optimized dashboard helper with caching"""
    _CACHE_TIMEOUT = 300  # 5 minutes
    
    @staticmethod
    def get_cached_data():
        """Get cached dashboard data with timeout"""
        cache_key = 'dashboard_data'
        cached = getattr(app, f'_cache_{cache_key}', None)
        
        if cached and (datetime.now() - cached['timestamp']).seconds < DashboardHelper._CACHE_TIMEOUT:
            return cached['data']
        
        # Calculate fresh data
        try:
            data = DashboardHelper._load_recent_data()
            avg_data = DashboardHelper._calculate_monthly_averages(data)
            
            # Update cache
            setattr(app, f'_cache_{cache_key}', {
                'data': avg_data,
                'timestamp': datetime.now()
            })
            
            return avg_data
        except Exception as e:
            app.logger.error(f"Failed to get cached data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _load_recent_data(months=5):
        """Optimized database query for recent data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30*months)
            query = db.session.query(
                LibericaBeanMetadata.created_at,
                LibericaBeanMetadata.area,
                LibericaBeanMetadata.perimeter,
                LibericaBeanMetadata.equivalent_diameter,
                LibericaBeanMetadata.extent,
                LibericaBeanMetadata.axis_major_length,
                LibericaBeanMetadata.axis_minor_length,
                LibericaBeanMetadata.eccentricity,
                LibericaBeanMetadata.class_label
            ).filter(LibericaBeanMetadata.created_at >= cutoff_date)
            
            return pd.read_sql(query.statement, db.engine)
        except Exception as e:
            app.logger.error(f"Failed to load recent data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _calculate_monthly_averages(data):
        """Vectorized monthly average calculation"""
        try:
            if data.empty:
                return pd.DataFrame()
                
            data['created_at'] = pd.to_datetime(data['created_at'])
            data['month_year'] = data['created_at'].dt.to_period('M')
            
            avg_data = data.groupby(['month_year', 'class_label']).agg({
                'area': 'mean',
                'perimeter': 'mean',
                'equivalent_diameter': 'mean',
                'extent': 'mean',
                'axis_major_length': 'mean',
                'axis_minor_length': 'mean',
                'eccentricity': 'mean'
            }).reset_index()
            
            avg_data['created_at'] = avg_data['month_year'].dt.strftime('%Y-%m')
            return avg_data.drop(columns=['month_year'])
        except Exception as e:
            app.logger.error(f"Failed to calculate monthly averages: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def create_graph(avg_data, selected_feature, class_label):
        """Optimized graph creation with Plotly"""
        try:
            if not selected_feature or not class_label or avg_data.empty:
                return go.Figure().to_html(full_html=False)
            
            filtered_data = avg_data[avg_data['class_label'] == class_label]
            if filtered_data.empty:
                return go.Figure().to_html(full_html=False)
            
            fig = go.Figure()
            if selected_feature == 'Area':
                for feature in ['area', 'convex_area']:
                    if feature in filtered_data.columns:
                        fig.add_trace(go.Bar(
                            x=filtered_data['created_at'],
                            y=filtered_data[feature],
                            name=feature.replace('_', ' ').title()
                        ))
            else:
                feature_lower = selected_feature.lower()
                if feature_lower in filtered_data.columns:
                    fig.add_trace(go.Bar(
                        x=filtered_data['created_at'],
                        y=filtered_data[feature_lower],
                        name=selected_feature,
                        marker_color='#591f0b'
                    ))
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=filtered_data['created_at'],
                    ticktext=filtered_data['created_at']
                ),
                yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            app.logger.error(f"Failed to create graph: {str(e)}")
            return go.Figure().to_html(full_html=False)

# --------------------------
# Flask Routes (Optimized)
# --------------------------

@app.route('/')
def homepage():
    """Optimized homepage with template caching"""
    try:
        return render_template("homepage.html")
    except Exception as e:
        app.logger.error(f"Error rendering homepage: {traceback.format_exc()}")
        flash("Error loading homepage", "error")
        return render_template("error.html"), 500

@app.route('/test-db')
def test_db():
    """Test database connection"""
    try:
        # Test simple database operation
        count = LibericaBeanMetadata.query.count()
        return f"Database connection successful. Records: {count}"
    except Exception as e:
        return f"Database error: {str(e)}<br>{traceback.format_exc()}", 500

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Optimized dashboard with cached data"""
    try:
        avg_data = DashboardHelper.get_cached_data()
        
        selected_feature = request.form.get('feature', 'area').capitalize()
        class_label = request.form.get('class_label', avg_data['class_label'].iloc[0] if len(avg_data) > 0 else '')
        
        graph_div = DashboardHelper.create_graph(avg_data, selected_feature, class_label)
        class_labels = avg_data['class_label'].unique().tolist() if len(avg_data) > 0 else []
        
        return render_template(
            "dashboard.html",
            graph_div=graph_div,
            features=['Area', 'Perimeter', 'Equivalent_diameter', 'Extent', 
                     'Axis_major_length', 'Axis_minor_length', 'Eccentricity'],
            selected_feature=selected_feature,
            class_label=class_label,
            class_labels=class_labels,
            title=f'Liberica Bean Analysis - {selected_feature} ({class_label})'
        )
    except Exception as e:
        app.logger.error(f"Error in dashboard route: {traceback.format_exc()}")
        flash("Error loading dashboard", "error")
        return render_template("error.html"), 500

@app.route('/records')
def records():
    """Optimized records view with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        sort_by = request.args.get('sort_by', 'id')
        sort_order = request.args.get('sort_order', 'asc')
        class_label = request.args.get('class_label')
        
        # Validate sort column
        valid_columns = [col.name for col in LibericaBeanMetadata.__table__.columns]
        sort_by = sort_by if sort_by in valid_columns else 'id'
        
        # Build base query
        query = LibericaBeanMetadata.query
        
        # Apply filters
        if class_label:
            query = query.filter_by(class_label=class_label)
        
        # Apply sorting
        sort_column = getattr(LibericaBeanMetadata, sort_by)
        query = query.order_by(sort_column.asc() if sort_order == 'asc' else sort_column.desc())
        
        # Paginate results
        pagination = query.paginate(
            page=page,
            per_page=Config.RECORDS_PER_PAGE,
            error_out=False
        )
        
        # Get unique class labels
        class_labels = db.session.query(
            LibericaBeanMetadata.class_label
        ).distinct().all()
        
        return render_template(
            'records.html',
            data=pagination.items,
            pagination=pagination,
            sort_by=sort_by,
            sort_order=sort_order,
            class_labels=[label[0] for label in class_labels],
            selected_class=class_label
        )
    except Exception as e:
        app.logger.error(f"Error in records route: {traceback.format_exc()}")
        flash("Error loading records", "error")
        return render_template("error.html"), 500

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    """Optimized scan endpoint with async processing"""
    try:
        if request.method == 'GET':
            return render_template('scan_coffee_bean.html')
        
        if 'file' not in request.files:
            flash("No file uploaded", "error")
            return render_template('scan_coffee_bean.html'), 400
        
        file = request.files['file']
        class_label = request.form.get('class_label', 'unknown')
        
        if not file or file.filename == '':
            flash("Invalid file", "error")
            return render_template('scan_coffee_bean.html'), 400
        
        analyzer = CoffeeBeanAnalyzer()
        future = analyzer.process_uploaded_file(file, class_label)
        
        try:
            df = future.result(timeout=120)  # 2 minute timeout
            if df is None or df.empty:
                flash("No beans detected in the image", "warning")
                return render_template('scan_coffee_bean.html')
            return render_template('results.html', df=df)
        except Exception as e:
            app.logger.error(f"Error processing file: {traceback.format_exc()}")
            flash("Error processing image", "error")
            return render_template('scan_coffee_bean.html'), 500
    except Exception as e:
        app.logger.error(f"Error in scan route: {traceback.format_exc()}")
        flash("Error processing request", "error")
        return render_template('error.html'), 500

# --------------------------
# Error Handlers
# --------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500

# --------------------------
# Application Startup
# --------------------------

@contextmanager
def app_context():
    """Context manager for application setup"""
    with app.app_context():
        yield

def initialize_app():
    """Optimized application initialization"""
    try:
        with app_context():
            # Verify database directory exists
            db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Create tables with optimized settings
            db.engine.execute("PRAGMA journal_mode=WAL")
            db.engine.execute("PRAGMA synchronous=NORMAL")
            db.engine.execute("PRAGMA cache_size=-10000")  # 10MB cache
            
            # Create tables if they don't exist
            db.create_all()
            
            # Additional performance optimizations
            db.engine.execute("PRAGMA optimize")
            print("Database initialized successfully")
    except Exception as e:
        print(f"Failed to initialize application: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    try:
        initialize_app()
        app.run(debug=True, threaded=True)
    except Exception as e:
        print(f"Application failed to start: {traceback.format_exc()}")
