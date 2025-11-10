from PIL import Image
import rasterio
import argparse
import matplotlib.pyplot as plt
from pyproj import Proj, transform
def pix2lonlat(x_pixel,y_pixel,transform_affine):
    lon,lat=rasterio.transform.xy(transform_affine,y_pixel,x_pixel)
    utm_proj = Proj(init='epsg:9001')  # UTM 投影
    latlon_proj = Proj(init='epsg:4326')  # WGS84 经纬度
    longitude, latitude = transform(utm_proj, latlon_proj, lon, lat)
    return longitude,latitude
def visual_img(tiff_path,label_path,use_latlon):
    # 打开 TIFF 文件
    with rasterio.open(tiff_path) as dataset:
        # 读取图像数据
        img = dataset.read()
        transform_affine = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height    
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))    
    axes[0].imshow(img.transpose(1,2,0),cmap='gray')
    axes[0].set_title('TIFF origin Image')
    axes[0].axis('off')
    x_mid=width//2
    y_mid=height//2
    lon,lat=pix2lonlat(x_mid,y_mid,transform_affine)
    print(transform_affine)
    print(lon,lat)
    if use_latlon:
        axes[0].text(0.5,-0.03,f"({lon:.4f}, {lat:.4f})",size=12,ha="center",va="center",transform=axes[0].transAxes)
    axes[1].imshow(img.transpose(1,2,0),cmap='gray')
    # 读取标签文件并打印内容
    with open(label_path, 'r') as label_file:
        for raw in label_file:
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue
            box_w = bw * width
            box_h = bh * height
            center_x = xc * width
            center_y = yc * height
            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0

            rect = plt.Rectangle((x1, y1), box_w, box_h, edgecolor='red', facecolor='none', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].set_title('TIFF Image with Label')
            axes[1].axis('off')

    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tiff_path",type=str,help="path to tiff image")
    parser.add_argument("--label_path",type=str,help="path to label file")
    parser.add_argument("--use_latlon",action="store_true",help="whether use lat lon or not")
    args=parser.parse_args()
    if args.label_path==None:
        args.label_path=args.tiff_path.replace("images","labels").replace(".tiff",".txt")   
    visual_img(tiff_path=args.tiff_path,label_path=args.label_path,use_latlon=args.use_latlon)