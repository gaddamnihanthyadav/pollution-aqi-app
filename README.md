# AI-Based Air Quality Prediction App

This is a Streamlit web application that uses AI/ML techniques to predict Air Quality Index (AQI), perform spatial interpolation for hyperlocal AQI estimation, suggest safe routes, and visualize air quality on a map.

## Features

- Generates synthetic air quality monitoring data
- Trains an XGBoost model to predict AQI
- Performs IDW interpolation for hyperlocal AQI estimation
- Suggests safe routes on a grid using predicted AQI values
- Visualizes AQI predictions and route on an interactive Folium map
- Easy-to-use Streamlit interface

## Setup & Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**
    ```bash
    streamlit run pollution.app.py
    ```

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies.

## File Structure

- `pollution.app.py` - Main Streamlit application file
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation (this file)

## Usage

1. After starting the app, open the provided local URL in your browser.
2. Click on **🚀 Run AQI Prediction Demo** to see the predictions, safe route, and AQI map.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.ai/)
- [Folium](https://python-visualization.github.io/folium/)
- [NetworkX](https://networkx.org/)

---

*This app uses synthetic data for demonstration purposes only.*
