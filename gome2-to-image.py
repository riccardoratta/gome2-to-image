import os
import os.path as p
import argparse
import numpy as np
import itertools
import rasterio

from netCDF4 import Dataset
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate
from shapely.ops import split
from multiprocessing import Pool
from rasterio.features import rasterize
from rasterio.transform import from_origin
from time import time

products = ["O3", "NO2", "NO2Tropo", "BrO", "SO2", "H2O", "HCHO"]

products_scale = {
    products[0]: 1,
    products[1]: 15,
    products[2]: 15,
    products[3]: 13,
    products[4]: 1,
    products[5]: 1,
    products[6]: 15,
}

parser = argparse.ArgumentParser(
    description="Convert GOME-2 NO2 data into GeoTIFF files",
)

parser.add_argument(
    "input_path",
    type=str,
    help="path of GOME-2 HDF5 file or a directory full of HDF5 files",
)

parser.add_argument(
    "--scale",
    type=int,
    default=0,
    help="multiplier to scale down the data, e.g., if scale is 15 each value will be multiplied by 1e-15",
)

parser.add_argument(
    "--error-threshold",
    type=float,
    default=0.7,
    help="set the error threshold (percentage between 0 and 1), to exclude invalid data, if 0 all positive values are used",
)

parser.add_argument("--output-dir", type=str)

parser.add_argument(
    "-p",
    "--product",
    type=str,
    choices=products,
    help="product to extract",
)

precision = np.float64

# boundary line on the east limit of WGS84 CRS
boundary = LineString([(180, -90), (180, 90)])

# output image size
image_size = (500, 1000)

# transform to align GOME-2 data
transform_40x80 = from_origin(-180, 90, 0.36, 0.36)


def error(message: str, code=1):
    print(message)
    exit(code)


def geolocation_iter(dataset):
    """
    Given a GOME-2 dataset load the pixels' geolocation points into an iterable.
    """
    v = dataset["GEOLOCATION"]

    lat = "Latitude"
    lng = "Longitude"

    return zip(
        v[lat + "A"][:],
        v[lng + "A"][:],
        v[lat + "C"][:],
        v[lng + "C"][:],
        v[lat + "D"][:],
        v[lng + "D"][:],
        v[lat + "B"][:],
        v[lng + "B"][:],
    )


def load_stripe(dataset_path: str, product: str, value_scale: float):
    """
    Given a GOME-2 netCDF4 dataset path, return the values and the associated errors into a list of
    polygons in the correct CRS.
    """
    dataset = Dataset(dataset_path)

    values = (dataset["TOTAL_COLUMNS"][product][:] * value_scale).tolist()

    values_error = (
        dataset["TOTAL_COLUMNS"][f"{product}_Error"][:] * value_scale
    ).tolist()

    polygons = []

    for i, lat_lng in enumerate(geolocation_iter(dataset)):
        lat_a, lng_a = lat_lng[0], lat_lng[1]
        lat_c, lng_c = lat_lng[2], lat_lng[3]
        lat_d, lng_d = lat_lng[4], lat_lng[5]
        lat_b, lng_b = lat_lng[6], lat_lng[7]

        max_lng = max(lng_a, lng_b, lng_c, lng_d)
        min_lng = min(lng_a, lng_b, lng_c, lng_d)

        diff = max_lng - min_lng

        # fix few strange coords
        if diff > 180:
            if lng_a > 180:
                lng_a -= 360
            if lng_b > 180:
                lng_b -= 360
            if lng_c > 180:
                lng_c -= 360
            if lng_d > 180:
                lng_d -= 360

        polygon = Polygon(
            [
                (lng_a, lat_a),
                (lng_c, lat_c),
                (lng_d, lat_d),
                (lng_b, lat_b),
            ]
        )

        if max_lng <= 180:
            # polygon is ALL before 180, ok
            polygons.append(polygon)
        elif min_lng > 180:
            # polygon is ALL after 180, translate by -360
            polygons.append(translate(polygon, -360))
        else:
            # polygon in the middle, split in two polygons at 180 degrees
            splits = split(polygon, boundary).geoms
            for j, sub_polygon in enumerate(splits):
                if max(sub_polygon.bounds) <= 180:
                    polygons.append(sub_polygon)
                else:
                    polygons.append(translate(sub_polygon, -360))
                # duplicate value at i if at least there are 2 sub-polygons
                if j != 0:
                    values.insert(i, values[i])
                    values_error.insert(i, values_error[i])

    dataset.close()

    min_l = min(len(values), len(polygons))

    values = values[0:min_l]
    polygons = polygons[0:min_l]

    return (polygons, values, values_error)


def decuple(value):
    """
    Given a list of tuples with 3 element, decuple the second and third element while keeping fixed
    the first. I.e., given a list of [(1, 2, 3), (2, 3, 4)], we get
    [(1, 2), (1, 3), (2, 3), (2, 4)].
    """
    return list(itertools.chain(*map(lambda x: [(x[0], x[1]), (x[0], x[2])], value)))


def rasterize_values(geometries, values):
    """
    Given a list of geometries and values, generate a raster matrix, for more information see the
    `rasterize` function of rasterio.
    """
    return rasterize(
        shapes=zip(geometries, values),
        out_shape=image_size,
        transform=transform_40x80,
    )


def commond_string(strings: list[str]):
    """
    Return the initial common part of a list of strings.
    """
    n = 0

    for i, characters in enumerate(zip(*strings)):
        if len(set(characters)) != 1:
            n = i
            break

    return strings[0][0:n]


if __name__ == "__main__":
    args = parser.parse_args()

    input_files = []

    if not p.exists(args.input_path):
        error(f"Input path does not exist")

    if args.output_dir is not None:
        if not p.exists(args.output_dir):
            os.makedirs(args.output_dir)

    if p.isdir(args.input_path):
        input_files = sorted(
            [
                p.join(args.input_path, path)
                for path in os.listdir(args.input_path)
                if path.endswith(".HDF5")
            ]
        )
    else:
        input_files = [p.basename(args.input_path)]

    print(f"Processing {len(input_files)} files")

    if args.scale != products_scale[args.product]:
        print(
            f"Warning: scale {args.scale} is not recommended for product {args.product} ({products_scale[args.product]})"
        )

    value_scale = 1 / 10**args.scale

    print(f"GOME2 data will be scaled by {value_scale}")

    start = time()

    print(f"Importing files.. ", end="")

    stripes = []

    for i, path in enumerate(input_files):
        stripes.append(
            load_stripe(
                p.join(p.dirname(args.input_path), path), args.product, value_scale
            )
        )
        print(f"\rImporting files.. [{i}/{len(input_files)}]", end="")

    print(f"\rImported {len(input_files)} files in {time() - start:.02f}s")

    start = time()

    print(f"Rasterizing stripes.. ", end="")

    rasters = []

    with Pool() as pool:
        rasters = pool.starmap(rasterize_values, decuple(stripes))

    print(f"\rRasterization completed in {time() - start:.02f}s")

    sum_value = np.zeros(image_size, dtype=precision)

    count = np.zeros(image_size, dtype=precision)

    for val, err in zip(rasters[0::2], rasters[1::2]):
        if args.error_threshold > 0:
            # compute the error percent with the value
            err_percent = np.divide(err, val, where=val != 0)
            # do not copy values with too much error..
            val[err_percent > args.error_threshold] = 0
        # .. or negative values
        val[val < 0.0] = 0

        sum_value += val

        count[val != 0] += 1

    val = np.divide(sum_value, count, where=count != 0)

    val[count == 0] = -1

    ext = f".{args.product}.tif"

    if len(input_files) == 1:
        output_file = input_files[0].replace(".HDF5", ext)
    else:
        output_file = commond_string([p.basename(file) for file in input_files]) + ext

    if args.output_dir is not None:
        output_file = p.join(args.output_dir, output_file)

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=image_size[0],
        width=image_size[1],
        count=1,
        dtype=precision,
        crs="WGS84",
        transform=transform_40x80,
        nodata=-1,
    ) as dataset:
        dataset.write(val, 1)

    print(f"Output file '{output_file}' written")
