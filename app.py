from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import io, urllib, base64
import datacube
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import odc.algo
import plotly.io as pio

from geopy.geocoders import Nominatim


import warnings

warnings.filterwarnings("ignore")


import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objs as go
import plotly.offline as pyoff

matplotlib.use('Agg')

dc = datacube.Datacube(app="Flask_Text")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/type/<analysis_type>', methods=['POST'])
def analysis(analysis_type):
    if request.method=="POST":
        data = request.get_json()
        
        coordinates = data['coordinates']
        time_range = (data['fromdate'], data['todate'])
        study_area_lat = (coordinates[0][0], coordinates[1][0])
        study_area_lon = (coordinates[1][1], coordinates[2][1])

        try:
            dc = datacube.Datacube(app='water_change_analysis')

            ds = dc.load(product='s2a_sen2cor_granule',
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:4326',
                resolution=(-0.00027, 0.00027)
            )
           
            ds = odc.algo.to_f32(ds)
            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            elif analysis_type=="graph":
                ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
                evi = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))

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
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])

                data = [['day', 'month', 'year', 'dense_forest', 'open_forest', 'sparse_forest', 'forest', 'total']]
                print(dense_forest_mask.shape[0])
                for i in range(dense_forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    print(data_time)
                    new_data_time = data_time.split("-")
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
                    
                   
                df = pd.DataFrame(data[1:], columns=data[0])

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                
                X = df[["year", "month"]]
                y = df["dense_forest"]
                y2 = df["open_forest"]
                y3 = df["sparse_forest"]

                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict([[2024,5]])
                print(df,y_pred)

                rf_regressor2 = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y2)
                y_pred2 = rf_regressor.predict([[2024,5]])
                print(df,y_pred2)

                rf_regressor3 = RandomForestRegressor(n_estimators=100, random_state=101)
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
                    title='Dense Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)

                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                

                return jsonify({"plot": plot_json})

            else:
                return jsonify({"error": "Invalid type"})

            res_start = res.sel(time=slice(time_range[0], time_range[1])).min(dim='time')
            res_end = res.sel(time=slice(time_range[0], time_range[1])).max(dim='time')
            res_diff = res_end - res_start
            print(time_range)

            if analysis_type=="ndvi":
                title = 'Vegetation'
                cmap = 'RdYlBu'
            elif analysis_type=="ndwi":
                title = 'Water'
                cmap = 'RdBu'

            sub_res = res.isel(time=[0, -1])
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
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()

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