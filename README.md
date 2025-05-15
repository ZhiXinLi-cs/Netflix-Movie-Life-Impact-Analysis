# Netflix-Movie-Life-Impact-Analysis
###1.Project Title:Netflix Movie Life Impact Analysis

**ZhiXin Li**

# Mainboard  
#### Abstract  
This analysis examines the factors that contribute to highly impactful Netflix movies, defined as those recommended by at least 80% of viewers. Using data from 82 movies, a predictive model was developed to identify key drivers of share-worthy viewing experiences. Key findings include the influence of genre, release year, and viewer engagement metrics. Recommendations are provided for content and platform teams to enhance movie impact.  

#### Rationale  
Understanding what makes Netflix movies highly recommendable is critical for maximizing viewer satisfaction, retention, and word-of-mouth marketing. This research helps Netflix prioritize content investments and optimize platform features to amplify the reach and impact of its movies.  

#### Research Question  
What factors determine whether a Netflix movie is highly impactful, i.e., recommended by at least 80% of viewers?  

#### Data Sources  
- Dataset of 82 Netflix movies, tracking 14 features including genre, release year, average rating, number of reviews, and discovery method.  
- No missing values in the dataset.  

#### Methodology  
### 1. Data Preparation  
- **Target Variable Creation**:  
  Converted "Suggested to Friends/Family (Y/N %)" into binary `High_Impact` label (≥80% recommendation rate)  
- **Feature Engineering**:  
  - Extracted minute value from "Minute of Life-Changing Insight"  
  - Created `Main_Genre` by taking the first genre from slash-separated values  
  - Parsed percentage value from recommendation field  
- **Missing Value Handling**:  
  - Filled missing `Insight_Minute` with median values  
  - Replaced null categorical values with "Unknown"  

### 2. Feature Processing Pipeline  

ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ], numeric_features),
        ('cat', 'passthrough', categorical_features)
    ])
)
- **Numerical Features**: Standardized after median imputation  
- **Categorical Features**: Label encoded while preserving original categories  

### 3. Feature Selection  
- **ANOVA F-test (f_classif)**: Selected top 5 most statistically significant features  
- **Selected Features**:  
  1. `Average Rating`  
  2. `Number of Reviews`  
  3. `Release Year`  
  4. `Main_Genre`  
  5. `How Discovered`  

### 4. Model Development  
- **Algorithm**: Random Forest Classifier  
- **Configuration**:  
  - 100 trees with balanced class weights  
  - 30% test set with stratified sampling  
  - Random state fixed (42) for reproducibility  

### 5. Evaluation Metrics  
- **Primary Metrics**:  
  - Accuracy  
  - AUC-ROC Score  
- **Diagnostic Tools**:  
  - Confusion Matrix  
  - Feature Importance Plot  
  - ROC Curve  
  - Genre Distribution Analysis  

### Key Technical Decisions  
1. **Switched from chi2 to f_classif** for better handling of continuous features  
2. **Added robust exception handling** for data loading and model saving  
3. **Visualization Suite** combining multiple diagnostic plots in a single figure  

#### Results  
- **Impact Distribution**: 31.7% of movies qualified as high-impact.  
- **Top Genres**: Comedy, Thriller, Documentary, Animation, and Drama were most common among high-impact movies.  
- **Model Performance**:  
  - Accuracy: 68% in predicting high-impact movies.  
  - Strong predictive power (AUC: 0.86).  
  - Better at identifying low-impact movies (88% correct) than high-impact ones (25% correct).  
- **Key Factors**:  
  1. Release Year (newer movies perform better).  
  2. Average Rating (higher ratings correlate with more recommendations).  
  3. Number of Reviews (more discussions increase spread).  
  4. Main Genre (certain genres outperform others).  
  5. Discovery Method (how viewers find the movie matters).  

#### Next Steps  
1. **Model Enhancements**:  
   - Address the imbalance in high-impact predictions.  
   - Experiment with alternative algorithms.  
   - Incorporate review text analysis.  
2. **Data Expansion**:  
   - Increase dataset size.  
   - Add viewer demographics and streaming metrics.  
3. **Implementation**:  
   - Develop an "Impact Predictor" dashboard.  
   - Create A/B testing frameworks for recommendations.  
   - Monitor emerging high-impact content in real-time.  

#### Conclusion  
The study successfully identified key factors driving high-impact Netflix movies, though the model has room for improvement in predicting true positives. Recommendations include focusing on high-performing genres, newer releases, and optimizing discovery methods. Future work should expand the dataset and refine the model for better accuracy.  

#### Bibliography  
(References would be listed here in Chicago style - ANS/AIP or AOV/IEE format.)  

#### Contact and Further Information  
Project Resources
Model Files: Available at netflix_impact_model.pkl (includes trained model, preprocessor, and feature selection objects)

Visualizations: Saved as netflix_impact_analysis.png (feature importance, confusion matrix, ROC curve, and genre distribution)

Dataset: Netflix Life Impact Dataset (NLID).csv (82 movies with 14 features)
For additional details, model files, or visualizations, please contact [ZhiXin Li] at [zli241@hawk.iit.edu].
