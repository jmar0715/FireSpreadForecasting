from sklearn import preprocessing
import fsspec
import numpy as np
import xarray as xr


def data_generator(file_list, batch_size, lag=2, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['Fuels', 'GPM.LATE.v5_FWI','GEOS-5_FWI','isPerimeter','isFireline']
    for fileObj in file_list:
        ds = None
        all_data = []
        print(fileObj)
        try:
            print(f"Processing file: {fileObj}")
            with fsspec.open(fileObj, mode="rb") as f:
                ds = xr.open_dataset(f, decode_coords='all', engine='netcdf4')
                ds.load()
                for col in feature_cols:
                    all_na = np.isnan(ds[col].values).all()
                    if all_na == True:
                        input_data = ds[col].transpose('x', 'y', 'time', 'channels')
                        input_data = input_data.isel(channels=0).fillna(0)
                    else:
                        input_data = ds[col].dropna(dim='channels', how='all')
                        input_data = input_data.transpose('x', 'y', 'time', 'channels')
                        input_data = input_data.fillna(0).isel(channels=0)

                    all_data.append(input_data)

                all_data = xr.concat(all_data, dim='channels')
                all_data = all_data.transpose('x', 'y', 'time', 'channels')
                all_data = np.array(all_data)

        except Exception as e1:
            print(e1)
            try:
                with fsspec.open(fileObj, mode="rb") as f:
                    ds = xr.open_dataset(f, decode_coords='all', engine='h5netcdf')
                    ds.load()
                    for col in feature_cols:
                        all_na = np.isnan(ds[col].values).all()
                        if all_na == True:
                            input_data = ds[col].transpose('x', 'y', 'time', 'channels')
                            input_data = input_data.isel(channels=0).fillna(0)
                        else:
                            input_data = ds[col].dropna(dim='channels', how='all')
                            input_data = input_data.transpose('x', 'y', 'time', 'channels')
                            input_data = input_data.fillna(0).isel(channels=0)

                        all_data.append(input_data)

                    all_data = xr.concat(all_data, dim='channels')

                    all_data = all_data.transpose('x', 'y', 'time', 'channels')

                    all_data = np.array(all_data)

            except Exception as e2:
                print(e2)
                # raise Exception(e2)
                continue
                
        time_len = len(ds.time)
        max_idx = time_len - lag

        for i in range(0, max_idx, batch_size):
            if i + batch_size <= max_idx:
                batch_data = all_data[:, :, i:i + batch_size, :]
            else:
                batch_data = all_data[:, :, i:max_idx, :]

            if lag + batch_size + i <= time_len:
                target_data = all_data[:, :, lag + i:lag + i + batch_size, -2:]
            else:
                target_data = all_data[:, :, lag + i:time_len, -2:]

            mean = np.mean(batch_data, axis=(0, 1, 2), keepdims=True)
            std = np.std(batch_data, axis=(0, 1, 2), keepdims=True)
            epsilon = 1e-8
            std = np.where(std == 0, epsilon, std)
            normalized_batch = (batch_data - mean) / std

            # target_data = target_data.values

            print(f"Yielding batch_data shape: {normalized_batch.shape}, target_data shape: {target_data.shape}")

            yield normalized_batch, target_data
