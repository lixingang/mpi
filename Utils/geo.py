from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import time
import os
import sys
import shutil
import numpy as np

def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.2f' %
              (func.__name__, time.time() - local_time))
    return wrapper


def get_mutil_geom(src):
    ds = ogr.Open(src, 0)
    if ds is None:
        sys.exit('Could not open {0}.'.format(src))

    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

    lyr = ds.GetLayer(0)  # 获得第一个图层，一般矢量只有一个图层
    for fea in lyr:  # 遍历图层的所有要素
        pt = fea.geometry()  # 获得要素的几何图形
        multipolygon.AddGeometry(pt)
    ds.Destroy()
    return multipolygon


@print_run_time
def shift_shp(src_path, dst_path, distance, field_name):
    ds = ogr.Open(src_path, 0)
    if ds is None:
        sys.exit('Could not open {0}.'.format(src_path))

    lyr = ds.GetLayer(0)
    proj = lyr.GetSpatialRef()

    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    dst_ds = oDriver.CreateDataSource(dst_path)
    dst_layer = dst_ds.CreateLayer('shift', geom_type=ogr.wkbPolygon, srs=proj)
    oFieldID = ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(oFieldID, 1)
    olayer_Defn = dst_layer.GetLayerDefn()

    for fea in lyr:  # 遍历图层的所有要素
        shift_geom = ogr.Geometry(ogr.wkbPolygon)
        pt = fea.geometry()  # 获得要素的几何图形
        field_value = fea.GetField(field_name)
        num = pt.GetGeometryCount()
        for i in range(num):
            line = pt.GetGeometryRef(i)
            shift_line = ogr.Geometry(ogr.wkbLinearRing)
            point_count = line.GetPointCount()
            for j in range(point_count):
                x = line.GetX(j) + distance
                y = line.GetY(j) + distance
                shift_line.AddPoint(x, y)
            shift_geom.AddGeometry(shift_line)

        oFeature = ogr.Feature(olayer_Defn)
        oFeature.SetGeometry(shift_geom)
        oFeature.SetField(0, field_value)
        dst_layer.CreateFeature(oFeature)
    dst_ds.Destroy()
    ds.Destroy()


def delete_dir(path):
    shutil.rmtree(path)
    if os.path.exists(path):
        os.rmdir(path)
# 计算时间函数


class ReadTif(object):
    def __init__(self, path, read_only=True, get_data=True):
        if not isinstance(path, str):
            raise RuntimeError('TIF文件路径错误')
        if not (path.endswith('.tif') or path.endswith('.TIF')):
            raise RuntimeError('请输入tif格式图像')
        if read_only:
            mode = gdal.GA_ReadOnly
        else:
            mode = gdal.GA_Update

        temp_time = time.time()
        self.read_only = read_only
        gdal.AllRegister()
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        dataset = gdal.Open(path, mode)
        self.im_width = dataset.RasterXSize  # 栅格矩阵的列数
        self.im_height = dataset.RasterYSize  # 栅格矩阵的行数
        self.nbands = dataset.RasterCount
        self.geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        self.proj = dataset.GetProjection()  # 地图投影信息

#         print('current Function [%s] run time is %.2f' % ('GDAL.Open', time.time() - temp_time))
        if get_data:
            self.im_data = dataset.ReadAsArray(
                0, 0, self.im_width, self.im_height)  # 将数据写成数组，对应栅格矩阵

        """
        获得的图像shape为：(c, h, w)
        """
        if read_only:
            del dataset
        else:
            self.dataset = dataset

    @print_run_time
    def set_data(self, data, index=0):
        '''
        data: 格式为(c, h, w)多通道或者为(h,w)单通道
        index: 单通道图像对应int数字， 多通道图像为list/tuple，长度需和通道数保持一致; 任一索引值不得超过总波段数
        index从0开始
        '''
        if data.ndim == 2 and isinstance(index, int):
            if index >= self.nbands:
                raise RuntimeError('error：插入索引应小于总波段数！')
            band = self.dataset.GetRasterBand(index+1)  # 将数据写入第index+1个波段
            band.WriteArray(data, 0, 0)
            band.FlushCache()
        elif data.ndim == 3 and type(index) in (tuple, list):
            if data.shape[0] != len(index):
                raise RuntimeError('error：索引个数应和输入数据通道数保持一致！')
            for i in range(len(index)):
                ind = index[i]
                if not isinstance(ind, int):
                    raise RuntimeError('error：插入索引应为int格式！')
                if ind >= self.nbands:
                    raise RuntimeError('error：插入索引应小于总波段数！')
                band = self.dataset.GetRasterBand(ind+1)
                band.WriteArray(data[i], 0, 0)
                band.FlushCache()

    def del_data(self):
        del self.im_data

    def set_proj(self, proj):
        if self.read_only:
            raise RuntimeError('文件为只读模式，禁止写入内容！')
        try:
            self.dataset.SetProjection(proj)
        except:
            print('warning: 投影设置错误！')

    def set_geotrans(self, geotrans=None):
        '''
        六个参数分别是：
        geos[0]  top left x 左上角x坐标
        geos[1]  w-e pixel resolution 东西方向像素分辨率
        geos[2]  rotation, 0 if image is "north up" 旋转角度，正北向上时为0
        geos[3]  top left y 左上角y坐标
        geos[4]  rotation, 0 if image is "north up" 旋转角度，正北向上时为0
        geos[5]  n-s pixel resolution 南北向像素分辨率
        x/y为图像的x/y坐标，geox/geoy为对应的投影坐标
        geox = geos[0] + geos[1] * x + geos[2] * y;
        geoy = geos[3] + geos[4] * x + geos[5] * y
        '''

        if self.read_only:
            raise RuntimeError('文件为只读模式，禁止写入内容！')

        if geotrans == None:
            raise RuntimeError('请输入仿射参数')
            # 示例： geotrans = [12014182,1.0,0,3679041,0,-1.0]
        if type(geotrans) not in (tuple, list):
            raise RuntimeError('仿射参数错误，无法设置')
        if len(geotrans) != 6:
            raise RuntimeError('仿射参数必须为6个！')

        self.dataset.SetGeoTransform(geotrans)

    def set_wgs84(self):
        if self.read_only:
            raise RuntimeError('文件为只读模式，禁止写入内容！')
        proj = osr.SpatialReference()
        proj.SetWellKnownGeogCS("WGS84")
        wgs84 = proj.ExportToWkt()
        self.dataset.SetProjection(wgs84)

    def set_webM(self):
        if self.read_only:
            raise RuntimeError('文件为只读模式，禁止写入内容！')
        proj_webM = 'PROJCS[\
                    "WGS_1984_Web_Mercator_Auxiliary_Sphere",\
                    GEOGCS[\
                   "GCS_WGS_1984",\
                   DATUM["D_WGS_1984", SPHEROID["WGS_1984", 6378137.0, 298.257223563]],\
                   PRIMEM["Greenwich", 0.0],\
                   UNIT["Degree", 0.0174532925199433]],\
                    PROJECTION["Mercator_Auxiliary_Sphere"],\
                    PARAMETER["False_Easting", 0.0],\
                    PARAMETER["False_Northing", 0.0],\
                    PARAMETER["Central_Meridian", 0.0],\
                    PARAMETER["Standard_Parallel_1", 0.0],\
                    PARAMETER["Auxiliary_Sphere_Type", 0.0],\
                    UNIT["Meter", 1.0]]'
        self.dataset.SetProjection(proj_webM)

    def close_dataset(self):
        if not self.read_only:
            self.dataset.FlushCache()
            del self.dataset


class SaveTif(object):
    def __init__(self, path, c, h, w, data_type='byte', proj=None, transform=None):
        if not isinstance(path, str):
            raise RuntimeError('error：保存路径需要为str格式！')
        if not (path.endswith('.tif') or path.endswith('.TIF')):
            raise RuntimeError('error：请输入tif格式路径')

        if data_type == 'byte':
            dtype = gdal.GDT_Byte
        elif data_type == 'float32':
            dtype = gdal.GDT_Float32
        else:
            raise RuntimeError(f'error：不支持的文件格式:{type}，请自行扩展！')

        gdal.AllRegister()
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        driver = gdal.GetDriverByName("GTiff")  # tif数据驱动，创建tif文件
        self.dataset = driver.Create(path, w, h, c, dtype)
        self.nbands = c
        if proj != None:
            if not isinstance(proj, str):
                raise RuntimeError('error：投影信息需要为str格式！')
            try:
                self.dataset.SetProjection(proj)  # 地图投影信息
            except:
                print('warning：投影设置错误！')

        if transform != None:
            if type(transform) not in (list, tuple):
                raise RuntimeError('error：仿射信息输入错误！')
            try:
                self.dataset.SetGeoTransform(transform)  # 地图仿射变换参数
            except:
                print('warning：仿射参数设置错误！')

    def set_proj(self, proj):
        try:
            self.dataset.SetProjection(proj)
        except:
            print('warning: 投影设置错误！')

    def set_geotrans(self, geotrans):
        '''
        六个参数分别是：
        geos[0]  top left x 左上角x坐标
        geos[1]  w-e pixel resolution 东西方向像素分辨率
        geos[2]  rotation, 0 if image is "north up" 旋转角度，正北向上时为0
        geos[3]  top left y 左上角y坐标
        geos[4]  rotation, 0 if image is "north up" 旋转角度，正北向上时为0
        geos[5]  n-s pixel resolution 南北向像素分辨率
        x/y为图像的x/y坐标，geox/geoy为对应的投影坐标
        geox = geos[0] + geos[1] * x + geos[2] * y;
        geoy = geos[3] + geos[4] * x + geos[5] * y
        '''
        # 示例： geotrans = [12014182,1.0,0,3679041,0,-1.0]

        if type(geotrans) not in (tuple, list):
            raise RuntimeError('仿射参数错误，无法设置')
        if len(geotrans) != 6:
            raise RuntimeError('仿射参数必须为6个！')

        self.dataset.SetGeoTransform(geotrans)

    def set_data(self, data, index=0):
        '''
        data: 格式为(c, h, w)多通道或者为(h,w)单通道
        index: 单通道图像对应int数字， 多通道图像为list/tuple，长度需和通道数保持一致; 任一索引值不得超过总波段数
        index从0开始
        '''
        if data.ndim == 2 and isinstance(index, int):
            if index >= self.nbands:
                raise RuntimeError('error：插入索引应小于总波段数！')
            band = self.dataset.GetRasterBand(index+1)  # 将数据写入第index+1个波段
            band.WriteArray(data, 0, 0)
            band.FlushCache()
        elif data.ndim == 3 and type(index) in (tuple, list):
            if data.shape[0] != len(index):
                raise RuntimeError('error：索引个数应和输入数据通道数保持一致！')
            for i in range(len(index)):
                ind = index[i]
                if not isinstance(ind, int):
                    raise RuntimeError('error：插入索引应为int格式！')
                if ind >= self.nbands:
                    raise RuntimeError('error：插入索引应小于总波段数！')
                band = self.dataset.GetRasterBand(ind+1)
                band.WriteArray(data[i], 0, 0)
                band.FlushCache()
        self.dataset.FlushCache()

    def set_wgs84(self):
        proj = osr.SpatialReference()
        proj.SetWellKnownGeogCS("WGS84")
        wgs84 = proj.ExportToWkt()
        self.dataset.SetProjection(wgs84)

    def set_webM(self):
        proj_webM = 'PROJCS[\
                    "WGS_1984_Web_Mercator_Auxiliary_Sphere",\
                    GEOGCS[\
                   "GCS_WGS_1984",\
                   DATUM["D_WGS_1984", SPHEROID["WGS_1984", 6378137.0, 298.257223563]],\
                   PRIMEM["Greenwich", 0.0],\
                   UNIT["Degree", 0.0174532925199433]],\
                    PROJECTION["Mercator_Auxiliary_Sphere"],\
                    PARAMETER["False_Easting", 0.0],\
                    PARAMETER["False_Northing", 0.0],\
                    PARAMETER["Central_Meridian", 0.0],\
                    PARAMETER["Standard_Parallel_1", 0.0],\
                    PARAMETER["Auxiliary_Sphere_Type", 0.0],\
                    UNIT["Meter", 1.0]]'
        self.dataset.SetProjection(proj_webM)

    def close_dataset(self):
        self.dataset.FlushCache()
        del self.dataset

def gdal_read(path ):
    dataset = gdal.Open(path)
    info = {}
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_band = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    info['bands'] = im_band
    info['width'] = im_width
    info['height'] = im_height
    info['geotrans'] = im_geotrans
    info['proj'] = im_proj
#     info['path'] = path
    return im_data, info

def writeTiff(im_data, width, height, bands, geotrans, proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
 
    if len(im_data.shape) == 3:
        bands, height, width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        bands, (height, width) = 1, im_data.shape
#     print(bands, height, width)
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, width, height, bands, datatype, ['COMPRESS=LZW', 'BIGTIFF=YES'])
    if (dataset != None):
        dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
        dataset.SetProjection(proj)  # 写入投影
    for i in range(bands):
#         print(im_data[i].shape)
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def split_img(img_path, split_number, save_dir):
    img, info = gdal_read(img_path)
    print(info)
    src_img_width = info['width']
    src_img_height = info['height']
    new_img_width = int(src_img_width /split_number)
    new_img_height = int(src_img_height /split_number)
    #得到左上角的坐标
    coor_y = [ i*new_img_height for i in range(split_number)]
    coor_x = [ j*new_img_width for j in range(split_number)]
    gt = info['geotrans']
    for i in coor_y:
        for j in coor_x:
            if (j+new_img_width > src_img_width) | (i+new_img_height>new_img_height):
                #x向越界
                if (j+new_img_width > src_img_width) & (i+new_img_height<=new_img_height):
                    data  = img[i:i+new_img_height,j:src_img_width]
 
                #y向越界
                elif (j+new_img_width <= src_img_width) & (i+new_img_height>new_img_height):
                    data = img[i:src_img_height,j:j+new_img_width]
 
                #xy方向均越界
                else:
                    #print(cols-i*size,rows-j*size)
                    data= img[j:src_img_height,i:new_img_width]
 
            else:
                data = img[i:i+new_img_height, j:j+new_img_width]
 
            new_gt  = (gt[0] + j  * gt[1], gt[1], gt[2], gt[3] + i * gt[5], gt[4], gt[5])
            info['geotrans'] = new_gt
            new_height, new_wight = data.shape
            info['width'] = new_wight
            info['height'] = new_height
            info['path'] = os.path.join(save_dir, 
                                       os.path.basename(img_path)[:-4] + "_%s_%s"%(str(i),str(j))+'.tif')
#             print(data.shape, info)
#             plt.imshow(data)
#             plt.show()
            writeTiff(data,**info)
            print('write %s'%info['path'])