import warnings 
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, redirect, jsonify # webframe work used to build web applicarions
from datetime import datetime # library used to work with dates and times
import io, urllib, base64 # library used to encode and decode images
import datacube # library used to manage and analyze large amounts of geospatial data
import matplotlib.pyplot as plt # used for data visualisation
import matplotlib
import matplotlib.gridspec as gridspec
import pandas as pd # used for data manipulation and analysis
from sklearn.model_selection import train_test_split # used for ML analysis regression and classification
from sklearn.ensemble import RandomForestRegressor
import numpy as np # used for mathematical operations
import odc.algo # used for processing satellite data
import plotly.io as pio # used for interactive data visualisation
from geopy.geocoders import Nominatim # used for geocoding and reverse geocoding 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

matplotlib.use('Agg') # indicate the backend to be used by Matplotlib

dc = datacube.Datacube(app="Flask_Text")
app = Flask(__name__)

# rendering index page 

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

# based on the analysis type the output is rendered accordingly

@app.route('/type/<analysis_type>', methods=['POST'])
def analysis(analysis_type):
    if request.method=="POST":
        # converting the data we got into json format
        data = request.get_json()

        coordinates = data['coordinates'] # getting coordinates from the form
        time_range = (data['fromdate'], data['todate']) # getting to and from date from form
        study_area_lat = (coordinates[0][0], coordinates[1][0]) # lat_range from coordinates
        study_area_lon = (coordinates[1][1], coordinates[2][1]) # lon_range from coordinates

        try:
            dc = datacube.Datacube(app='water_change_analysis')

            # creating a dataset object (product,lon_range,lat_range,time,bands)
            ds = dc.load(product='s2a_sen2cor_granule',
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:4326',
                resolution=(-0.00027, 0.00027)
            )
            
            ds = odc.algo.to_f32(ds) # converting everything into floating values,ML oftens requies float values for analysis

            # based on analysis the formula is derived

            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            elif analysis_type=="graph": #if its graph we need random forest analysis code to compute.
                ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
                evi = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))

                # fixing a threshold value for ndvi and evi to segregate forest type
                ndvi_threshold = 0.4
                evi_threshold = 0.2

                forest_mask_ndvi = np.where(ndvi > ndvi_threshold, 1, 0)
                forest_mask_evi = np.where(evi > evi_threshold, 1, 0)

                forest = np.logical_and(forest_mask_ndvi, forest_mask_evi)

                # Create forest masks based on NDVI and EVI thresholds
                dense_forest_mask = np.where((ndvi > ndvi_threshold) & (evi > evi_threshold), 1, 0)
                open_forest_mask = np.where((ndvi > ndvi_threshold) & (evi <= evi_threshold), 1, 0)
                sparse_forest_mask = np.where((ndvi <= ndvi_threshold) & (evi <= evi_threshold), 1, 0)

                # Calculate the area of each pixel
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4]) # to calculate pixel area

                data = [['day', 'month', 'year', 'dense_forest', 'open_forest', 'sparse_forest', 'forest', 'total']]
                print(dense_forest_mask.shape[0])
                for i in range(dense_forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    print(data_time)
                    new_data_time = data_time.split("-") #splitting year month date
                    # Calculate the forest cover area for each forest type
                    dense_forest_cover_area = np.sum(dense_forest_mask[i]) * pixel_area
                    open_forest_cover_area = np.sum(open_forest_mask[i]) * pixel_area
                    sparse_forest_cover_area = np.sum(sparse_forest_mask[i]) * pixel_area
                    # Calculate the total forest cover area
                    total_forest_cover_area = dense_forest_cover_area + open_forest_cover_area + sparse_forest_cover_area
                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area
                    
                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                dense_forest_cover_area, open_forest_cover_area,
                                sparse_forest_cover_area, total_forest_cover_area, original])
                    
                   
                df = pd.DataFrame(data[1:], columns=data[0]) # converting data into dataframe 

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                
                X = df[["year", "month"]]
                y = df["dense_forest"]
                y2 = df["open_forest"]
                y3 = df["sparse_forest"]

                # Random Forest Model and plotting graphs according to data
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict([[2024,5]])
                print(df,y_pred)

                
                rf_regressor.fit(X, y2)
                y_pred2 = rf_regressor.predict([[2024,5]])
                print(df,y_pred2)

                
                rf_regressor.fit(X, y3)
                y_pred3 = rf_regressor.predict([[2024,5]])
                print(df,y_pred3)


                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')

                # Plot monthly forest
                plot_data = [
                go.Scatter(
                    x = df['year-month'],
                    y = df['dense_forest']/1000000,
                    name = "Dense Actual"
                ),
                go.Scatter(
                    x = ['2024-05'],
                    y = y_pred/1000000,
                    name = "Dense Predicted"
                ),
                go.Scatter(
                    x = df['year-month'],
                    y = df['open_forest']/1000000,
                    name = "Open Actual"
                ),
                go.Scatter(
                    x = ['2024-05'],
                    y = y_pred2/1000000,
                    name = "Open Predicted"
                ),
                go.Scatter(
                    x = df['year-month'],
                    y = df['sparse_forest']/1000000,
                    name = "Sparse Actual"
                ),
                go.Scatter(
                    x = ['2024-05'],
                    y = y_pred3/1000000,
                    name = "Sparse Predicted"
                ),
                ]

                print("Plot plotted")

                plot_layout = go.Layout(
                    title='Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout) #using plotly to visualize the data

                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                return jsonify({"plot": plot_json})

            else:
                return jsonify({"error": "Invalid type"})
            
            # For NDVI or NDWI
            res_start = res.sel(time=slice(time_range[0], time_range[1])).min(dim='time')
            res_end = res.sel(time=slice(time_range[0], time_range[1])).max(dim='time')
            print(time_range)

            if analysis_type=="ndvi":
                title = 'Vegetation'
                cmap = 'RdYlBu'
            elif analysis_type=="ndwi":
                title = 'Water'
                cmap = 'RdBu'

            sub_res = res.isel(time=[0, -1]) # select first and last timestamps of data
            mean_res = res.mean(dim=['latitude', 'longitude'], skipna=True)
            mean_res_rounded = list(map(lambda x: round(x, 4), mean_res.values.tolist()))
            labels = list(map(lambda x: x.split('T')[0], [i for i in np.datetime_as_string(res.time.values).tolist()])) 

            plot = sub_res.plot(col='time', col_wrap=2)
            for ax, time in zip(plot.axes.flat, res.time.values):
                ax.set_title(str(time).split('T')[0])

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)

            plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1,2)

            plt.subplot(gs[0,0])
            plt.imshow(res_start, cmap=cmap, vmin=-1, vmax=1)
            plt.title(title+' '+data['fromdate'][:4])

            plt.subplot(gs[0,1])
            plt.imshow(res_end, cmap=cmap, vmin=-1, vmax=1)
            plt.title(title+' '+data['todate'][:4])

            plt.colorbar()

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO() #stream object to stores bytes in memory
            plt.savefig(img, format='png') #save the plot to bytes object
            img.seek(0) #read the cursor from start
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) #encode bytes code and covert to url safe string
            plt.clf() # to clear the plot

            # to get area name of the tile selected 
            area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
            print(area_name)
            
            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates,"area_name":area_name,"type": analysis_type, "mean_res_rounded": mean_res_rounded, "labels": labels})
        except Exception as e:
            return jsonify({"error": e})
    return jsonify({"error": "Invalid method: "+request.method})

def get_area_name(latitude, longitude):
    geolocator = Nominatim(user_agent='my-app')  # Initialize the geocoder
    location = geolocator.reverse((latitude, longitude))  # Reverse geocode the coordinates
    if location is not None:
        address_components = location.raw['address']
        city_name = address_components.get('city', '')
        if not city_name:
            city_name = address_components.get('town', '')
        if not city_name:
            city_name = address_components.get('village', '')
        return city_name
    else:
        return "City name not found"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')