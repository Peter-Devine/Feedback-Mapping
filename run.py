from data.downloader import get_data

from mapping.mapper import map_data

get_data(["guzman_2015"], random_state=111)

map_data(["guzman_2015"], ["USE"])
