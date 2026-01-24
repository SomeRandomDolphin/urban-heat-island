import folium

bounds = {
    "min_lon": 106.6,
    "max_lon": 107.1,
    "min_lat": -6.4,
    "max_lat": -6.0
}

# Center of the box
center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Draw bounding box
folium.Rectangle(
    bounds=[
        [bounds["min_lat"], bounds["min_lon"]],
        [bounds["max_lat"], bounds["max_lon"]],
    ],
    color="red",
    fill=True,
    fill_opacity=0.2
).add_to(m)

m.save("bounding_box_map.html")