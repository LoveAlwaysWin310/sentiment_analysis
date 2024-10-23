import pandas as pd
import folium
from folium.plugins import HeatMap
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

# 加载数据
file_path = 'music_comments.csv'
column_names = ['username', 'user_id', 'region', 'comment', 'comment_id', 'unknown_number', 'timestamp']
data = pd.read_csv(file_path, names=column_names, header=None, skiprows=1)

# 统计每个地区的评论数
region_counts = data['region'].value_counts().reset_index()
region_counts.columns = ['region', 'count']

# 更新的地区到经纬度映射，包括更多地区
region_to_coords = {
    '广东': [23.1317, 113.2663], '江苏': [32.0617, 118.7778], '四川': [30.657, 104.066],
    '浙江': [30.267, 120.153], '湖南': [28.1127, 112.9834], '北京': [39.9042, 116.4074],
    '上海': [31.2304, 121.4737], '天津': [39.3434, 117.3616], '重庆': [29.5630, 106.5516],
    '河北': [38.0455, 114.5215], '山西': [37.8695, 112.5489], '辽宁': [41.8057, 123.4315],
    '吉林': [43.8378, 126.5494], '黑龙江': [45.7424, 126.6616], '安徽': [31.8612, 117.2858],
    '福建': [26.0753, 119.3062], '江西': [28.6743, 115.8922], '山东': [36.6683, 116.9972],
    '河南': [34.7657, 113.7532], '湖北': [30.5931, 114.3054], '广西': [22.8155, 108.3275],
    '甘肃': [36.0611, 103.8343], '陕西': [34.2649, 108.9542], '云南': [25.0406, 102.7123],
    '贵州': [26.5982, 106.7074], '西藏': [29.6480, 91.1172], '青海': [36.6171, 101.7782],
    '宁夏': [38.4713, 106.2587], '新疆': [43.7930, 87.6278], '内蒙古': [40.8175, 111.7656],
    '海南': [20.0199, 110.3487]
}

# 为统计数据添加经纬度
region_counts['latitude'] = region_counts['region'].apply(lambda x: region_to_coords.get(x, [None, None])[0])
region_counts['longitude'] = region_counts['region'].apply(lambda x: region_to_coords.get(x, [None, None])[1])

# 移除无法找到经纬度的行
region_counts_clean = region_counts.dropna()

# 准备热力图数据
heat_data = region_counts_clean[['latitude', 'longitude', 'count']].values.tolist()

# 绘制热力图
m = folium.Map(location=[34.2649, 108.9542], zoom_start=4)  # 调整了zoom_start以更好地展示全国范围
HeatMap(heat_data).add_to(m)
heatmap_file_path_complete = 'data/heatmap_with_more_regions.html'
m.save(heatmap_file_path_complete)

# 设置selenium webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 无头模式

# 指定chromedriver路径
s = Service('C:/Program Files/Google/Chrome/Application/chromedriver.exe')

# 使用Service对象创建Chrome WebDriver实例
driver = webdriver.Chrome(service=s)
driver.get(f'file:///C:/Users/Cherry/PycharmProjects/网易云音乐评论可视化展示与情感分析/data/heatmap_with_more_regions.html')
time.sleep(10)  # 等待地图加载

heatmap_image_path = 'picture/地区热力图'  # 更新为你想保存图片的路径
driver.save_screenshot(heatmap_image_path)

driver.quit()