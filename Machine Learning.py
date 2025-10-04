import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class StockPredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        try:
            df = pd.read_csv('final_data.csv', parse_dates=['Date'])
            print(f"Data loaded: {df.shape}")

            # Check if we have the split column
            if 'Split' in df.columns:
                train_data = df[df['Split'] == 'train']
                val_data = df[df['Split'] == 'valid']
                test_data = df[df['Split'] == 'test']

                print(f"Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}")
                return train_data, val_data, test_data
            else:
                # Manual split based on time
                df_sorted = df.sort_values('Date')
                total_size = len(df_sorted)
                train_size = int(0.6 * total_size)
                val_size = int(0.2 * total_size)

                train_data = df_sorted[:train_size]
                val_data = df_sorted[train_size:train_size + val_size]
                test_data = df_sorted[train_size + val_size:]

                print(f"Manual split - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
                return train_data, val_data, test_data

        except FileNotFoundError:
            print("Error: final_data.csv not found")
            return None, None, None

    def prepare_features_and_targets(self, train_data, val_data, test_data):
        """Prepare features and target variables"""
        # Define feature columns to exclude
        exclude_cols = ['Date', 'Target', 'Tomorrow_Close', 'Tomorrow_Return', 'Split', 'Sector']
        ticker_cols = [col for col in train_data.columns if col.startswith('Ticker_')]
        exclude_cols.extend(ticker_cols)

        # Get feature columns
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features for modeling")

        # Prepare features
        X_train = train_data[feature_cols]
        X_val = val_data[feature_cols]
        X_test = test_data[feature_cols]

        # Prepare targets
        y_train_clf = train_data['Target'] if 'Target' in train_data.columns else None
        y_val_clf = val_data['Target'] if 'Target' in val_data.columns else None
        y_test_clf = test_data['Target'] if 'Target' in test_data.columns else None

        y_train_reg = train_data['Tomorrow_Return'] if 'Tomorrow_Return' in train_data.columns else None
        y_val_reg = val_data['Tomorrow_Return'] if 'Tomorrow_Return' in val_data.columns else None
        y_test_reg = test_data['Tomorrow_Return'] if 'Tomorrow_Return' in test_data.columns else None

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
            'y_train_clf': y_train_clf, 'y_val_clf': y_val_clf, 'y_test_clf': y_test_clf,
            'y_train_reg': y_train_reg, 'y_val_reg': y_val_reg, 'y_test_reg': y_test_reg,
            'feature_names': feature_cols
        }

    def train_classification_models(self, data):
        """Train classification models for up/down prediction"""
        print("Training Classification Models...")

        X_train, y_train = data['X_train'], data['y_train_clf']
        X_val, y_val = data['X_val'], data['y_val_clf']

        if y_train is None:
            print("No target variable found for classification")
            return

        # Define models
        clf_models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision_Tree': DecisionTreeClassifier(random_state=42),
            'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
            'Neural_Network': MLPClassifier(random_state=42, max_iter=500)
        }

        # Train and evaluate models
        clf_results = {}
        for name, model in clf_models.items():
            print(f"Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            # Calculate metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val)

            clf_results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }

            print(f"{name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        self.models['classification'] = clf_results
        print("Classification models training completed")

    def train_regression_models(self, data):
        """Train regression models for return prediction"""
        print("Training Regression Models...")

        X_train, y_train = data['X_train'], data['y_train_reg']
        X_val, y_val = data['X_val'], data['y_val_reg']

        if y_train is None:
            print("No target variable found for regression")
            return

        # Define models
        reg_models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(random_state=42),
            'Lasso_Regression': Lasso(random_state=42),
            'Decision_Tree': DecisionTreeRegressor(random_state=42),
            'Random_Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
            'Neural_Network': MLPRegressor(random_state=42, max_iter=500)
        }

        # Train and evaluate models
        reg_results = {}
        for name, model in reg_models.items():
            print(f"Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)

            reg_results[name] = {
                'model': model,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2
            }

            print(f"{name} - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, R2: {val_r2:.4f}")

        self.models['regression'] = reg_results
        print("Regression models training completed")

    def hyperparameter_tuning(self, data, model_type='classification'):
        """Perform hyperparameter tuning for best models"""
        print(f"Starting hyperparameter tuning for {model_type}...")

        if model_type == 'classification' and 'classification' in self.models:
            # Select best classification model
            best_clf_name = max(self.models['classification'].keys(),
                                key=lambda x: self.models['classification'][x]['val_f1'])
            print(f"Tuning {best_clf_name}")

            X_train, y_train = data['X_train'], data['y_train_clf']

            if best_clf_name == 'Random_Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestClassifier(random_state=42)

            elif best_clf_name == 'Gradient_Boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                base_model = GradientBoostingClassifier(random_state=42)

            else:
                print(f"Hyperparameter tuning not configured for {best_clf_name}")
                return

            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(base_model, param_grid, cv=tscv,
                                       scoring='f1', n_jobs=-1, verbose=1)

            grid_search.fit(X_train, y_train)

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")

            self.best_model = grid_search.best_estimator_

        elif model_type == 'regression' and 'regression' in self.models:
            # Select best regression model
            best_reg_name = max(self.models['regression'].keys(),
                                key=lambda x: self.models['regression'][x]['val_r2'])
            print(f"Tuning {best_reg_name}")

            X_train, y_train = data['X_train'], data['y_train_reg']

            if best_reg_name == 'Random_Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestRegressor(random_state=42)

            elif best_reg_name == 'Gradient_Boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                base_model = GradientBoostingRegressor(random_state=42)

            else:
                print(f"Hyperparameter tuning not configured for {best_reg_name}")
                return

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(base_model, param_grid, cv=tscv,
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

            grid_search.fit(X_train, y_train)

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.6f}")

            self.best_model = grid_search.best_estimator_

    def evaluate_final_model(self, data, model_type='classification'):
        """Evaluate the best model on test set"""
        print("Evaluating final model on test set...")

        if self.best_model is None:
            print("No best model found. Please run hyperparameter tuning first.")
            return

        X_test = data['X_test']

        if model_type == 'classification':
            y_test = data['y_test_clf']
            y_pred = self.best_model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"Test Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()

        else:  # regression
            y_test = data['y_test_reg']
            y_pred = self.best_model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Test Results:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R2 Score: {r2:.4f}")

            # Prediction vs Actual plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            plt.title('Predicted vs Actual Returns')
            plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.show()

    def analyze_feature_importance(self, data):
        """Analyze feature importance"""
        if self.best_model is None:
            print("No model available for feature importance analysis")
            return

        feature_names = data['feature_names']

        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            print("Top 20 Most Important Features:")
            print(importance_df.head(20))

            # Plot feature importance
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            self.feature_importance = importance_df

        elif hasattr(self.best_model, 'coef_'):
            # For linear models
            importance = np.abs(self.best_model.coef_)
            if len(importance.shape) > 1:
                importance = importance[0]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            print("Top 20 Most Important Features (Linear Model):")
            print(importance_df.head(20))

            self.feature_importance = importance_df

    def generate_model_summary(self):
        """Generate comprehensive model summary"""
        print("\n" + "=" * 60)
        print("MODEL DEVELOPMENT SUMMARY")
        print("=" * 60)

        if 'classification' in self.models:
            print("\nClassification Results:")
            for name, results in self.models['classification'].items():
                print(f"{name:20s} - Val Accuracy: {results['val_accuracy']:.4f}, F1: {results['val_f1']:.4f}")

        if 'regression' in self.models:
            print("\nRegression Results:")
            for name, results in self.models['regression'].items():
                print(f"{name:20s} - Val R2: {results['val_r2']:.4f}, MSE: {results['val_mse']:.6f}")

        if self.best_model is not None:
            print(f"\nBest Model: {type(self.best_model).__name__}")
            print("Model is ready for deployment")

        print(f"\nGenerated files:")
        print("- confusion_matrix.png or prediction_vs_actual.png")
        print("- feature_importance.png")


def main():
    """Main execution function"""
    # Initialize model development
    stock_model = StockPredictionModel()

    # Load and prepare data
    train_data, val_data, test_data = stock_model.load_and_prepare_data()
    if train_data is None:
        return

    # Prepare features and targets
    data = stock_model.prepare_features_and_targets(train_data, val_data, test_data)

    # Train models
    stock_model.train_classification_models(data)
    stock_model.train_regression_models(data)

    # Hyperparameter tuning (choose classification or regression)
    model_type = 'classification'  # Change to 'regression' if needed
    stock_model.hyperparameter_tuning(data, model_type)

    # Evaluate final model
    stock_model.evaluate_final_model(data, model_type)

    # Analyze feature importance
    stock_model.analyze_feature_importance(data)

    # Generate summary
    stock_model.generate_model_summary()

    print("\nModel Development Completed!")


if __name__ == "__main__":
    main()


