Analyzing NO2 and SO2 feature stacks...

 NO2 Feature Stack Analysis:
   Total features: 33
   Feature names: ['no2_target', 'no2_mask', 'year', 'day', 'dem', 'slope', 'pop', 'lulc_class_0', 'lulc_class_1', 'lulc_class_2', 'lulc_class_3', 'lulc_class_4', 'lulc_class_5', 'lulc_class_6', 'lulc_class_7', 'lulc_class_8', 'lulc_class_9', 'sin_doy', 'cos_doy', 'weekday_weight', 'u10', 'v10', 'blh', 'tp', 't2m', 'sp', 'str', 'ssr_clr', 'ws', 'wd_sin', 'wd_cos', 'no2_lag_1day', 'no2_neighbor']
   Feature categories:
     target: ['no2_target']
     mask: ['no2_mask']
     metadata: ['year', 'day']
     static: ['dem', 'slope', 'pop']
     lulc: ['lulc_class_0', 'lulc_class_1', 'lulc_class_2', 'lulc_class_3', 'lulc_class_4', 'lulc_class_5', 'lulc_class_6', 'lulc_class_7', 'lulc_class_8', 'lulc_class_9']
     time: ['sin_doy', 'cos_doy', 'weekday_weight']
     meteo: ['u10', 'v10', 'blh', 'tp', 't2m', 'sp', 'str', 'ssr_clr']
     derived: ['ws', 'wd_sin', 'wd_cos']
     dynamic: ['no2_lag_1day', 'no2_neighbor']

 SO2 Feature Stack Analysis:
   Total features: 20
   Feature names: ['X', 'y', 'mask', 'feature_names', 'cont_idx', 'onehot_idx', 'noscale_idx', 'coverage', 'trainable', 'pollutant', 'season', 'date', 'doy', 'weekday', 'year_len', 'grid_height', 'grid_width', 'lag1_fill_ratio', 'neighbor_fill_ratio', 'file_version']
   X array feature names: ['dem', 'slope', 'population', 'lulc_class_10', 'lulc_class_20', 'lulc_class_30', 'lulc_class_40', 'lulc_class_50', 'lulc_class_60', 'lulc_class_70', 'lulc_class_80', 'lulc_class_90', 'lulc_class_100', 'u10', 'v10', 'ws', 'wd_sin', 'wd_cos', 'blh', 'tp', 't2m', 'sp', 'str', 'ssr_clear', 'so2_lag1', 'so2_neighbor', 'so2_climate_prior', 'sin_doy', 'cos_doy', 'weekday_weight']
   Feature categories:
     target: ['y']
     mask: ['mask']
     metadata: ['date', 'doy', 'weekday', 'year_len', 'grid_height', 'grid_width']
     static: ['dem', 'slope', 'population']
     lulc: ['lulc_class_10', 'lulc_class_20', 'lulc_class_30', 'lulc_class_40', 'lulc_class_50', 'lulc_class_60', 'lulc_class_70', 'lulc_class_80', 'lulc_class_90', 'lulc_class_100']
     time: ['sin_doy', 'cos_doy', 'weekday_weight']
     meteo: ['u10', 'v10', 'blh', 'tp', 't2m', 'sp', 'str', 'ssr_clear']
     derived: ['ws', 'wd_sin', 'wd_cos']
     dynamic: ['so2_lag1', 'so2_neighbor']
     special: ['so2_climate_prior']