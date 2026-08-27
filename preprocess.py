import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta 
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, chi2, RFE, SelectFromModel

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# XGBoost
from xgboost import XGBClassifier

# Model Selection & Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                           roc_curve, precision_recall_curve, average_precision_score, confusion_matrix,
                           classification_report, log_loss, matthews_corrcoef)

# Imbalanced Data
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from pathlib import Path
import sqlite3

class LeadPreprocessor:
    """
    Complete preprocessing pipeline for lead scoring model
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []  # Initialize as empty list instead of None
        self.is_fitted = False
        self.one_hot_columns = []
        self.top_categories = {}  # Store known categories for high cardinality features
        self.best_model = None
        
        # Initialize mappings
        self.journey_mapping = {
            'QnA about unit': 'QnA',
            'Selecting Room': 'Room_Selection',
            'Viewing Booked': 'Property_Viewing',
            'Info collection - Getting all question at once': 'Information_Collection',
            'Sales flow entered': 'Sales_flow',
            'Arranging Viewing': 'Viewing_Arrangement',
            'Processing': 'Processing',
            'Unknown': 'Unknown'
        }
        
        self.gender_mapping = {
            'Male': 'Male', 'M': 'Male', 'male': 'Male', 'MALE': 'Male',
            'Man': 'Male', 'Boy': 'Male',
            'Female': 'Female', 'F': 'Female', 'female': 'Female', 'FEMALE': 'Female',
            'Woman': 'Female', 'Girl': 'Female', 'Lady': 'Female',
            'Mix': 'Mixed', 'mix': 'Mixed', 'MIX': 'Mixed', 'Mixed': 'Mixed',
            'Multiple': 'Mixed', 'Both': 'Mixed', 'Couple': 'Mixed', 'Group': 'Mixed',
            'Unknown': 'Unknown', 'unknown': 'Unknown', 'UNKNOWN': 'Unknown',
            'Not Specified': 'Unknown', 'Not specified': 'Unknown',
            'N/A': 'Unknown', 'Na': 'Unknown', 'Prefer Not To Say': 'Unknown'
        }
        
        self.tenancy_period_mapping = {
            '12 months': '12', '12': '12',
            '6 months': '6', '6': '6',
            'Unknown': 'Unknown'
        }
    
    def standardize_capitalization(self, series):
        """Advanced standardization with proper capitalization rules"""
        # Convert to string and strip whitespace
        cleaned = series.astype(str).str.strip()

        # Handle NaN/null representations
        cleaned = cleaned.replace(['Nan', 'nan', 'NaN', 'None', 'none', '','Null'], 'Unknown')

        # Apply title case for most fields
        cleaned = cleaned.str.title()

        # Fix common capitalization issues
        replacements = {
            'Qna': 'QnA',
            'Fb': 'FB',
            'Ig': 'IG',
            'Wa': 'WA',
            'Whatsapp': 'WhatsApp',
            'Walkin': 'Walk-In',
            'Walk In': 'Walk-In',
            'Walk-in': 'Walk-In'
        }

        for old, new in replacements.items():
            cleaned = cleaned.str.replace(old, new, regex=False)

        return cleaned

    def standardize_nationality(self, nationality):
        """Standardize nationality to clean country format"""
        nat = str(nationality).lower().strip()

        # Handle obvious data entry errors first
        if nat in ['', '.', 'nan', 'none']:
            return 'Unknown'

        # Country mappings (nationality → country)
        nationality_mapping = {
            # Malaysia variations
            'malaysia': 'Malaysia',
            'malaysian': 'Malaysia',

            # Indonesia variations
            'indonesia': 'Indonesia',
            'indonesian': 'Indonesia',

            # India variations
            'india': 'India',
            'indian': 'India',
            'indian, residing in kuwait': 'India',
            # Sudan variations
            'sudan': 'Sudan',
            'sudanese': 'Sudan',
            'sudane': 'Sudan',  # Typo

            # Zimbabwe variations
            'zimbabwe': 'Zimbabwe',
            'zimbabwean': 'Zimbabwe',

            # Other countries
            'china': 'China',
            'thailand': 'Thailand',
            'thai': 'Thailand',
            'myanmar': 'Myanmar',
            'pakistan': 'Pakistan',
            'yemen': 'Yemen',
            'philippines': 'Philippines',
            'nigeria': 'Nigeria',
            'zambia, africa': 'Zambia',
            'sri lankan': 'Sri Lanka',
            'egyptian': 'Egypt',
            'congolese': 'Congo',
            'kenyan': 'Kenya',

            # Handle obvious errors
            'klang': 'Malaysia', 
            'attached bathroom': 'Unknown',
            'african': 'Unknown',  
            'kadazan': 'Malaysia',  

            # Unknown
            'unknown': 'Unknown'
        }

        if nat in nationality_mapping:
            return nationality_mapping[nat]

        return nationality.title()

    def group_location(self, location):
        """Group location search into categories"""
        if pd.isna(location):
            return 'Unknown'
        location = str(location).lower()
        if any(term in location for term in ['kl city', 'klcc', 'bukit bintang']):
            return 'KL_City'
        elif 'cheras' in location:
            return 'Cheras'
        elif any(term in location for term in ['mont kiara', 'hartamas']):
            return 'Mont_Kiara'
        elif 'university' in location or any(term in location for term in ['taylor', 'sunway']):
            return 'University_Area'
        elif 'hotel' in location:
            return 'Hotel_Area'
        elif any(term in location for term in ['petaling jaya', 'pj']):
            return 'Petaling_Jaya'
        elif 'subang' in location:
            return 'Subang'
        elif any(term in location for term in ['any', 'anywhere']):
            return 'Flexible'
        else:
            return 'Other'

    def create_binary_target(self, viewing_status):
        """Create binary target based on viewing status"""
        if pd.isna(viewing_status) or viewing_status is None:
            return 0  # Not Success
        elif viewing_status == 'Lose/Not Interested':
            return 0  # Not Success
        else:
            return 1  # Success

    def categorize_budget(self, budget):
        """Categorize budget into ranges"""
        if budget == 0:
            return 'Unknown'
        elif budget < 500:
            return 'Low'
        elif budget < 1000:
            return 'Medium'
        elif budget < 1500:
            return 'High'
        else:
            return 'Premium'

    def categorize_move_urgency(self, days):
        """Categorize move-in urgency based on days"""
        try:
            if pd.isna(days) or not isinstance(days, (int, float)):
                return 'Unknown'
            days_num = float(days)
            if days_num <= 30:
                return 'Urgent'
            elif days_num <= 90:
                return 'Soon'
            else:
                return 'Future'
        except:
            return 'Unknown'

    def create_primary_lead_source(self, row):
        """Create primary lead source with priority"""
        sources = [
            row.get('combined_lead_source', 'Unknown'),
            row.get('lead_source', 'Unknown'),
            row.get('source_from', 'Unknown')
        ]
        
        for source in sources:
            if source not in ['Unknown', 'Nan', ''] and pd.notna(source):
                return str(source).strip().title()
        return 'Unknown'

    def analyze_data(self, df):
        """Analyze the data structure"""
        print("Analyzing data structure...")
        print(f"Dataset shape: {df.shape}")
        
        # Check categorical columns
        categorical_columns = ['customer_journey', 'location_search', 'selected_property', 
                              'lead_source', 'gender', 'transportation', 'parking', 
                              'nationality', 'source_from', 'combined_lead_source', 
                              'room_type', 'viewing_status', 'contact_dayofweek', 'tenancy_period']
        
        for col in categorical_columns:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                print(f"{col}: {len(unique_vals)} unique values")
                if len(unique_vals) <= 10:
                    print(f"  Values: {list(unique_vals)}")
                else:
                    print(f"  Sample: {list(unique_vals[:5])}...")

    def fit(self, df):
        """Fit the preprocessing pipeline"""
        print("Fitting preprocessing pipeline...")
        
        # Analyze data first
        self.analyze_data(df)
        
        # Create a copy for preprocessing
        df_clean = df.drop(columns=["lead_score", "customer_id", "inserted_at","clean_phone"], errors="ignore").copy()
        
        # Create binary target
        df_clean['Target_Success'] = df_clean['viewing_status'].apply(self.create_binary_target)
        df_clean = df_clean.drop(columns=['viewing_status'], errors='ignore')
        # Handle dates safely
        for col in ['initial_contact_date', 'last_action_date', 'move_in_date']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Fill missing last_action_date with initial_contact_date
        if 'last_action_date' in df_clean.columns and 'initial_contact_date' in df_clean.columns:
            df_clean['last_action_date'] = df_clean['last_action_date'].fillna(df_clean['initial_contact_date'])
        
        # Handle missing values
        text_columns = [
            'customer_journey', 'location_search', 'selected_property', 'lead_source',
            'tenancy_period', 'gender', 'transportation', 'parking',
            'nationality', 'source_from', 'combined_lead_source', 'room_type', 'contact_dayofweek'
        ]
        
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown').astype(str)
        
        numerical_fill_columns = ['budget', 'no_of_pax', 'rental_proposed', 'contact_hour', 'contact_month', 'frequency', 'recencydays']
        for col in numerical_fill_columns:
            if col in df_clean.columns:
                if col in ['budget', 'no_of_pax']:
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Feature standardization
        text_fields_to_standardize = [
            'location_search', 'selected_property', 'lead_source', 'gender', 'transportation',
            'parking', 'nationality', 'source_from', 'combined_lead_source', 'room_type'
        ]
        
        for col in text_fields_to_standardize:
            if col in df_clean.columns:
                df_clean[col] = self.standardize_capitalization(df_clean[col])
        
        # Apply specific mappings
        df_clean['customer_journey'] = df_clean['customer_journey'].map(self.journey_mapping).fillna('Unknown')
        df_clean['location_search'] = df_clean['location_search'].apply(self.group_location)
        df_clean['Gender_Clean'] = df_clean['gender'].map(self.gender_mapping).fillna('Unknown')
        df_clean['tenancy_period'] = df_clean['tenancy_period'].map(self.tenancy_period_mapping).fillna('Unknown')
        df_clean['Nationality_Standard'] = df_clean['nationality'].apply(self.standardize_nationality)
        df_clean['Lead_Source_Primary'] = df_clean.apply(self.create_primary_lead_source, axis=1)
        
        # Feature engineering
        df_clean['budget'] = df_clean['budget'].apply(self.categorize_budget)
        df_clean['Is_Weekend'] = df_clean['contact_dayofweek'].isin(['Saturday', 'Sunday']).astype(int)
        df_clean['Is_Business_Hours'] = ((df_clean['contact_hour'] >= 9) & (df_clean['contact_hour'] <= 17)).astype(int)
        
        # Calculate Days_to_Move safely
        try:
            df_clean['Days_to_Move'] = (df_clean['move_in_date'] - df_clean['initial_contact_date']).dt.days
        except:
            print("Warning: Error calculating Days_to_Move, setting to 0")
            df_clean['Days_to_Move'] = 0
        
        df_clean['Move_Urgency'] = df_clean['Days_to_Move'].apply(self.categorize_move_urgency)
        
        # Label encoding for ordinal features
        ordinal_features = ['budget', 'Move_Urgency']
        for col in ordinal_features:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le
        
        # One-hot encoding for nominal features
        nominal_features = [
            'customer_journey', 'location_search', 'Gender_Clean', 'transportation',
            'parking', 'contact_dayofweek', 'tenancy_period'
        ]

        dfs_to_concat = []  # for collecting dummy DataFrames

        # Handle high-cardinality features (selected_property, room_type, Lead_Source_Primary)
        high_cardinality_features = ['selected_property', 'room_type', 'Lead_Source_Primary','Nationality_Standard']
        self.top_categories = {}  # Store top categories for each high cardinality feature

        for col in high_cardinality_features:
            if col in df_clean.columns:
                unique_count = df_clean[col].nunique()

                if unique_count > 10:
                    print(f"\nHandling high-cardinality feature: {col} ({unique_count} categories)")
                    
                    # Store top 10 categories for transform
                    self.top_categories[col] = set(df_clean[col].value_counts().head(10).index)
                    
                    # Create grouped version
                    df_clean[f'{col}_Grouped'] = df_clean[col].apply(
                        lambda x: x if x in self.top_categories[col] else 'Other'
                    )

                    # One-hot encode the grouped version
                    dummies = pd.get_dummies(df_clean[f'{col}_Grouped'], 
                                            prefix=f'{col}_Grouped', drop_first=False)
                    dfs_to_concat.append(dummies)

                    # Store column names for future reference
                    self.one_hot_columns.extend(dummies.columns)

                    print(f"  ✅ {col}: Grouped into {len(self.top_categories[col])+1} categories "
                        f"-> {len(dummies.columns)} dummy variables")
                else:
                    # If not too many categories, just treat as normal nominal feature
                    nominal_features.append(col)

        # Process other nominal features with one-hot encoding
        for col in nominal_features:
            if col in df_clean.columns:
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=False)
                dfs_to_concat.append(dummies)

        # Combine everything into final DataFrame
        df_processed = pd.concat([df_clean] + dfs_to_concat, axis=1)
        
        columns_to_drop = [
            'lead_source',
            'combined_lead_source',
            'source_from',
            'gender',
            'selected_property',
            'location_search',
            'initial_contact_date',
            'move_in_date',
            'last_action_date',
            'room_type',
            'transportation',
            'parking',
            'contact_dayofweek',
            'nationality',
            'customer_journey',
            'Lead_Source_Primary',
            'tenancy_period',
            'Nationality_Standard'
        ]
        
        intermediate_to_drop = [
            'Lead_Source_Primary_Grouped',  
            'Days_to_Move',
            'Nationality_Standard_Grouped',
            'Gender_Clean',
            'selected_property_Grouped',
            'room_type_Grouped'
        ]
        
        utility_to_drop = [
            'clean_phone',              
            'inserted_at',              
            'lead_score'    
        ]
        
        columns_to_drop = columns_to_drop + intermediate_to_drop + utility_to_drop
        existing_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        
        df_processed = df_processed.drop(columns=existing_to_drop)

        # Save feature names and mark pipeline as fitted
        self.feature_names = [col for col in df_processed.columns
                            if col not in ['Target_Success', 'feature_id']]
        self.is_fitted = True

        return df_processed

    
    def transform(self, df):
        """Transform data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        print("Transforming data...")
        
        # Apply the same preprocessing steps as in fit()
        df_clean = df.drop(columns=["lead_score", "customer_id", "inserted_at","clean_phone"], errors="ignore").copy()
        
        # Create binary target if viewing_status exists
        if 'viewing_status' in df_clean.columns:
            df_clean['Target_Success'] = df_clean['viewing_status'].apply(self.create_binary_target)
        
        # Handle dates
        for col in ['initial_contact_date', 'last_action_date', 'move_in_date']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Fill missing last_action_date with initial_contact_date
        if 'last_action_date' in df_clean.columns and 'initial_contact_date' in df_clean.columns:
            df_clean['last_action_date'] = df_clean['last_action_date'].fillna(df_clean['initial_contact_date'])
        
        # Handle missing values (same as fit)
        text_columns = [
            'customer_journey', 'location_search', 'selected_property', 'lead_source',
            'tenancy_period', 'gender', 'transportation', 'parking',
            'nationality', 'source_from', 'combined_lead_source', 'room_type', 'contact_dayofweek'
        ]
        
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown').astype(str)
        
        numerical_fill_columns = ['budget', 'no_of_pax', 'rental_proposed', 'contact_hour', 'contact_month', 'frequency', 'recencydays']
        for col in numerical_fill_columns:
            if col in df_clean.columns:
                if col in ['budget', 'no_of_pax']:
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Apply all transformations (same as fit)
        text_fields_to_standardize = [
            'location_search', 'selected_property', 'lead_source', 'gender', 'transportation',
            'parking', 'nationality', 'source_from', 'combined_lead_source', 'room_type'
        ]
        
        for col in text_fields_to_standardize:
            if col in df_clean.columns:
                df_clean[col] = self.standardize_capitalization(df_clean[col])
        
        # Apply mappings
        df_clean['customer_journey'] = df_clean['customer_journey'].map(self.journey_mapping).fillna('Unknown')
        df_clean['location_search'] = df_clean['location_search'].apply(self.group_location)
        df_clean['Gender_Clean'] = df_clean['gender'].map(self.gender_mapping).fillna('Unknown')
        df_clean['tenancy_period'] = df_clean['tenancy_period'].map(self.tenancy_period_mapping).fillna('Unknown')
        df_clean['Nationality_Standard'] = df_clean['nationality'].apply(self.standardize_nationality)
        df_clean['Lead_Source_Primary'] = df_clean.apply(self.create_primary_lead_source, axis=1)
        
        # Feature engineering
        df_clean['budget'] = df_clean['budget'].apply(self.categorize_budget)
        df_clean['Is_Weekend'] = df_clean['contact_dayofweek'].isin(['Saturday', 'Sunday']).astype(int)
        df_clean['Is_Business_Hours'] = ((df_clean['contact_hour'] >= 9) & (df_clean['contact_hour'] <= 17)).astype(int)
        
        # Calculate Days_to_Move safely
        try:
            df_clean['Days_to_Move'] = (df_clean['move_in_date'] - df_clean['initial_contact_date']).dt.days
        except:
            df_clean['Days_to_Move'] = 0
        
        df_clean['Move_Urgency'] = df_clean['Days_to_Move'].apply(self.categorize_move_urgency)
        
        # Label encoding for ordinal features
        ordinal_features = ['budget', 'Move_Urgency']
        for col in ordinal_features:
            if col in df_clean.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_clean[col] = df_clean[col].map(lambda x: le.transform([str(x)])[0]
                                                    if str(x) in le.classes_ else -1)

        
        # One-hot encoding for nominal features
        nominal_features = [
            'customer_journey', 'location_search', 'Gender_Clean', 'transportation',
            'parking', 'contact_dayofweek', 'tenancy_period'
        ]

        dfs_to_concat = []  # for collecting dummy DataFrames

        high_cardinality_features = ['selected_property', 'room_type', 'Lead_Source_Primary', 'Nationality_Standard']

        for col in high_cardinality_features:
            if col in df_clean.columns:
                if col in self.top_categories:  # Use stored categories from fit
                    print(f"\nHandling high-cardinality feature: {col}")
                    
                    # Map values using stored categories, anything not in top_categories becomes 'Other'
                    df_clean[f'{col}_Grouped'] = df_clean[col].apply(
                        lambda x: x if x in self.top_categories[col] else 'Other'
                    )

                    # One-hot encode with fixed set of columns from training
                    dummies = pd.get_dummies(df_clean[f'{col}_Grouped'], 
                                          prefix=f'{col}_Grouped', drop_first=False)
                    
                    # Ensure all expected columns exist
                    expected_columns = [f"{col}_Grouped_{cat}" for cat in list(self.top_categories[col]) + ['Other']]
                    for exp_col in expected_columns:
                        if exp_col not in dummies.columns:
                            dummies[exp_col] = 0
                    
                    # Keep only columns we know from training
                    dummies = dummies[expected_columns]
                    dfs_to_concat.append(dummies)

                    print(f"  ✅ {col}: Using {len(expected_columns)} categories from training")
                else:
                    # If not seen during fit, treat as normal nominal feature
                    nominal_features.append(col)

        # Process other nominal features with one-hot encoding
        for col in nominal_features:
            if col in df_clean.columns:
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=False)
                dfs_to_concat.append(dummies)

        # Combine everything into final DataFrame
        df_final = pd.concat([df_clean] + dfs_to_concat, axis=1)
        
        columns_to_drop = [
            'lead_source',
            'combined_lead_source',
            'source_from',
            'gender',
            'selected_property',
            'location_search',
            'initial_contact_date',
            'move_in_date',
            'last_action_date',
            'room_type',
            'transportation',
            'parking',
            'contact_dayofweek',
            'nationality',
            'customer_journey',
            'Lead_Source_Primary',
            'tenancy_period',
            'Nationality_Standard'
        ]
        
        intermediate_to_drop = [
            'Lead_Source_Primary_Grouped',  
            'Days_to_Move',
            'Nationality_Standard_Grouped',
            'Gender_Clean',
            'selected_property_Grouped',
            'room_type_Grouped'
        ]
        
        utility_to_drop = [
            'clean_phone',              
            'inserted_at',              
            'lead_score'    
        ]
        
        columns_to_drop = columns_to_drop + intermediate_to_drop + utility_to_drop
        existing_to_drop = [col for col in columns_to_drop if col in df_final.columns]
        
        df_final =df_final.drop(columns=existing_to_drop)
        
        # Convert remaining non-numeric columns using fitted encoders
        non_numeric_cols = df_final.select_dtypes(exclude=[np.number]).columns
        non_numeric_cols = [col for col in non_numeric_cols if col != 'Target_Success']
        
        for col in non_numeric_cols:
            encoder_key = f'remaining_{col}'
            if encoder_key in self.label_encoders:
                le = self.label_encoders[encoder_key]
                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except ValueError:
                        return 0
                df_final[col] = df_final[col].astype(str).fillna('Unknown').apply(safe_transform)
        
        # Ensure same features as training
        missing_features = [f for f in self.feature_names if f not in df_final.columns]
        if missing_features:
            print(f"Adding {len(missing_features)} missing features with default values")
            for feature in missing_features:
                df_final[feature] = 0

        # Select only training features
        df_final = df_final[self.feature_names + (['Target_Success'] if 'Target_Success' in df_final.columns else [])]

        print(f"Data transformed to shape: {df_final.shape}")
        return df_final

    def train_model(self, df_preprocessed):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Prepare the data
        feature_cols = [col for col in df_preprocessed.columns if col not in ['Target_Success', 'feature_id']]
        X = df_preprocessed[feature_cols]
        y = df_preprocessed['Target_Success']
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train with default parameters (best performing based on your original analysis)
        self.best_model = RandomForestClassifier(random_state=42, n_estimators=100)
        self.best_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return self.best_model
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Transform the data
        df_transformed = self.transform(df)
        
        # Prepare features
        feature_cols = [col for col in df_transformed.columns if col not in ['Target_Success', 'feature_id']]
        X = df_transformed[feature_cols]
        
        # Make predictions
        probabilities = self.best_model.predict_proba(X)[:, 1]
        predictions = self.best_model.predict(X)
        
        return predictions, probabilities
        
    def save_rf_model(self, filepath='RFM model/rf_model.pkl'):
        """Save the rf_model to a file"""
        if not self.is_fitted:
            raise ValueError("rf_model must be fitted before saving. Call fit() first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(exist_ok=True)
        
        # Save the entire rf_model object
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"rf_model saved to {filepath}")
        
    @classmethod
    def load_rf_model(cls, filepath='RFM model/rf_model.pkl'):
        """Load the rf_model from a file"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
            
        if not isinstance(preprocessor, cls):
            raise ValueError(f"Loaded object is not an instance of {cls.__name__}")
            
        return preprocessor


def main():
    """Main execution function - returns preprocessed dataframe"""
    print("CONVERTING YOUR ALPS RFM MODEL TO REUSABLE PKL FILES")
    print("=" * 60)
    
    try:
        # Connect to the database
        DB_PATH = Path("belive.db")
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Database file '{DB_PATH}' not found")
        
        conn = sqlite3.connect(DB_PATH)
        
        # Load the data
        print("Loading data from database...")
        df = pd.read_sql_query("SELECT * FROM feature_snapshots", conn)
        conn.close()
        
        print(f"Loaded {len(df)} records")
        
        # Create and fit the preprocessor
        preprocessor = LeadPreprocessor()
        df_preprocessed = preprocessor.fit(df)
        
        print("\nStarting model training...")
        # Train the model
        model = preprocessor.train_model(df_preprocessed)
        
        print("\nMaking predictions on training data...")
        # Make predictions on training data
        predictions, probabilities = preprocessor.predict(df)
        
        # Create final leads DataFrame with predictions
        final_leads = df.copy()
        final_leads['predicted_success'] = predictions
        final_leads['success_probability'] = probabilities
        final_leads['lead_score'] = (probabilities * 100).astype(int)
        
        # Create RFM model directory
        Path('RFM model').mkdir(exist_ok=True)
        
        # Save final leads with predictions
        final_leads.to_csv('RFM model/final_leads.csv', index=False)
        print("Saved final leads with predictions to RFM model/final_leads.csv")
        
        # Save the preprocessing pipeline
        preprocessor.save_rf_model()
        
        # Save feature importance if model is trained
        if preprocessor.best_model is not None and hasattr(preprocessor.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': preprocessor.feature_names,
                'importance': preprocessor.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv('RFM model/feature_importance.csv', index=False)
            print("Saved feature importance to RFM model/feature_importance.csv")
        
        # Create usage example
        usage_example = '''
# Example usage of the saved model
import pickle
import joblib
import pandas as pd
from pathlib import Path

# Load the rf_model
preprocessor = LeadPreprocessor.load_rf_model('RFM model/rf_model.pkl')

# Load new data
new_data = pd.read_csv("new_leads.csv")

# Make predictions
predictions, probabilities = preprocessor.predict(new_data)

# Add results to dataframe
new_data["predicted_success"] = predictions
new_data["success_probability"] = probabilities
new_data["lead_score"] = (probabilities * 100).astype(int)

print("Predictions completed!")
print(f"Average success probability: {probabilities.mean():.3f}")
print(f"Predicted success rate: {predictions.mean():.1%}")
'''
        
        print("\nSUCCESS! Model conversion completed!")
        print("\nFiles created in RFM model/ directory:")
        print("- rf_model.pkl (trained Random Forest model)")
        print("- feature_importance.csv (feature importance scores)")
        print("- final_leads.csv (original data with predictions)")
        
        # RETURN THE PREPROCESSED DATAFRAME
        return True
        
    except Exception as e:
        print(f"CONVERSION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Main execution
if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nModel successfully converted to reusable pickle files!")
        print("You can now use the saved pipeline for new predictions.")
    else:
        print("\nCONVERSION FAILED! Please check the error messages above.")