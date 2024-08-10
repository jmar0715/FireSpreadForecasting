import fsspec
import xarray as xr
import argparse


def main(input_file, output_file, columns):
    columns = columns.split(' ')
    print(columns)
    with fsspec.open(input_file) as f:
        ds = None
        engine = "h5netcdf"
        try:
            ds = xr.open_dataset(f, decode_coords='all', engine=engine)
        except:
            engine = "netcdf4"
            ds = xr.open_dataset(f, decode_coords='all', engine=engine)

        filtered_ds = ds[columns]
        filtered_ds.to_netcdf(path=output_file, engine=engine)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", required=True, help="Input file")
    ap.add_argument("-o", "--output_file", required=True, help="Output file")
    ap.add_argument("-c", "--columns", type=str, help="Space separated list of columns to keep",
                    default="Fuels GPM.LATE.v5_FWI GEOS-5_FWI isPerimeter isFireline")
    args = ap.parse_args()
    main(args.input_file, args.output_file, args.columns)
