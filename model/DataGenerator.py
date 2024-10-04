import fsspec
import numpy as np
import xarray as xr

def data_processing(engine_type, feature_cols, fileObj, batch_size=1, lag=2):
    all_data = []
    with fsspec.open(fileObj, mode="rb") as f:
        with xr.open_dataset(f, decode_coords='all', engine = engine_type, 
                            cache = True, chunks = 'auto') as ds: #setting chunks as arbitrary

            for col in feature_cols:
                num_non_na = ds[col].count().values.sum()
                if num_non_na == 0:
                    input_data = ds[col].isel(channels=0).fillna(0)
                else:
                    input_data = ds[col].dropna(dim='channels', how='all')
                    input_data = input_data.fillna(0).isel(channels=0)
                
                input_data = input_data.expand_dims(dim='channels', axis=-1)
                input_data = input_data.transpose('x', 'y', 'time', 'channels')
                
                input_data = input_data.chunk({'x': input_data.sizes['x'], 
                                               'y': input_data.sizes['y'], 
                                               'time': input_data.sizes['time'], 
                                               'channels': 1}) #1 chunk per input.. 5 input
                
                all_data.append(input_data)
                
            all_data_concat = xr.concat(all_data, dim='channels')
            
            all_data_concat = all_data_concat.chunk({'x': all_data_concat.sizes['x'], 
                                                     'y': all_data_concat.sizes['y'], 
                                                     'time': all_data_concat.sizes['time'],
                                                    'channels': 5})
            
            all_data_concat = all_data_concat.compute()

            time_len = all_data_concat.sizes['time']
            max_idx = time_len - lag

            for i in range(0, max_idx, batch_size):
                if i + batch_size <= max_idx:
                    batch_data = all_data_concat.isel(time = slice(i,i+batch_size))
                else:
                    batch_data = all_data_concat.isel(time = slice(i,max_idx))

                if lag + batch_size + i <= time_len:
                    target_data = all_data_concat.isel(time = slice(lag + i,lag + i + batch_size), channels=slice(-2,None))
                else:
                    target_data = all_data_concat.isel(time = slice(lag + i, time_len), channels = slice(-1,None))
                
                mean = np.mean(batch_data, axis=2, keepdims=True)
                std = np.std(batch_data, axis=2, keepdims=True)
                epsilon = 1e-8
                std = np.where(std == 0, epsilon, std)
                normalized_batch = (batch_data - mean) / std

                # target_data = target_data.values

                print(f"Yielding batch_data shape: {normalized_batch.shape}, target_data shape: {target_data.shape}")

                yield normalized_batch, target_data


def data_generator(file_list, batch_size, lag=2, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['Fuels', 'GPM.LATE.v5_FWI','GEOS-5_FWI','isPerimeter','isFireline']
    for fileObj in file_list:
        ds = None
        all_data = []
        print(fileObj)
        try:
            print(f"Processing file: {fileObj}")
            yield from data_processing('h5netcdf', feature_cols, fileObj, batch_size, lag=2)

        except Exception as e1:
            print(e1)
            try:
                yield from data_processing('netcdf4', feature_cols, fileObj, batch_size, lag=2)

            except Exception as e2:
                print(e2)
                continue
