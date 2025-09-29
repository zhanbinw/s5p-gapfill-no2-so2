# Feature Engineering Implementation

## NO2 Feature Stack Generation Code

### Overview
This document contains the core implementation code for generating NO2 feature stacks from the 17 variables identified in the feature engineering plan.

### Batch Generation Script

**File**: `batch_generate_no2_fixed_colab.py`

```python
#!/usr/bin/env python3
"""
NO2特征栈批量生成脚本（修复版，Colab版本）
Batch NO2 Feature Stack Generation (Fixed Version, Colab Version)

使用修复后的代码批量生成2019-2023年的NO2特征栈，支持断点续跑
"""

import os
import numpy as np
import sys
from datetime import datetime, timedelta
import calendar
import time

# Colab兼容：不使用__file__，直接导入
try:
    from no2_independent_feature_stack_builder_fixed import NO2IndependentFeatureStackBuilder
except ImportError:
    print("❌ 无法导入NO2IndependentFeatureStackBuilder")
    print("💡 请确保no2_independent_feature_stack_builder_fixed.py文件在正确位置")
    sys.exit(1)

class NO2BatchGenerator:
    """NO2特征栈批量生成器"""
    
    def __init__(self, base_path: str = "/content/drive/MyDrive"):
        """初始化批量生成器"""
        self.base_path = base_path
        self.builder = NO2IndependentFeatureStackBuilder(base_path)
        self.output_dir = self.builder.output_dir
        
        # 创建进度跟踪文件
        self.progress_file = os.path.join(self.output_dir, "batch_progress.txt")
        
        print(f"🚀 NO2 Batch Generator initialized")
        print(f"   - Base path: {self.base_path}")
        print(f"   - Output directory: {self.output_dir}")
        print(f"   - Progress file: {self.progress_file}")
    
    def generate_year(self, year: int, resume: bool = True) -> bool:
        """生成单年特征栈"""
        print(f"\n📅 Generating NO2 feature stacks for {year}")
        print("=" * 50)
        
        # 检查数据可用性
        if not self.builder._check_data_availability(year):
            print(f"❌ Data availability check failed for {year}")
            return False
        
        # 加载静态特征（一次性加载，避免重复）
        print("📊 Loading static features...")
        static_features = self.builder._load_static_features()
        lulc_features = self.builder._load_lulc_features()
        aoi_mask = self.builder._load_aoi_mask()
        temporal_features = self.builder._load_temporal_features(year)
        
        # 获取该年的天数
        total_days = 366 if calendar.isleap(year) else 365
        print(f"📊 Total days in {year}: {total_days}")
        
        # 检查已完成的文件
        completed_files = set()
        if resume:
            completed_files = self._get_completed_files(year)
            print(f"📊 Already completed: {len(completed_files)} files")
        
        # 生成每日特征栈
        success_count = 0
        start_time = time.time()
        
        for day in range(1, total_days + 1):
            # 检查是否已完成
            target_date = datetime(year, 1, 1) + timedelta(days=day-1)
            date_str = target_date.strftime("%Y%m%d")
            output_file = os.path.join(self.output_dir, f"NO2_stack_{date_str}.npz")
            
            if resume and date_str in completed_files:
                success_count += 1
                if day % 50 == 0:
                    print(f"   ⏭️ Day {day:03d} ({date_str}): Already completed, skipping")
                continue
            
            try:
                # 构建单日特征栈
                daily_stack = self.builder._build_daily_no2_stack(
                    year, day, static_features, lulc_features, temporal_features, aoi_mask
                )
                
                if daily_stack is not None:
                    # 保存特征栈
                    np.savez_compressed(output_file, **daily_stack)
                    success_count += 1
                    
                    # 更新进度
                    self._update_progress(year, day, date_str, True)
                    
                    if day % 50 == 0 or day <= 10:
                        elapsed = time.time() - start_time
                        rate = day / elapsed if elapsed > 0 else 0
                        eta = (total_days - day) / rate if rate > 0 else 0
                        print(f"   ✅ Day {day:03d} ({date_str}): Generated, rate: {rate:.1f} files/min, ETA: {eta/60:.1f} min")
                else:
                    self._update_progress(year, day, date_str, False)
                    print(f"   ❌ Day {day:03d} ({date_str}): Failed to generate")
                
            except Exception as e:
                self._update_progress(year, day, date_str, False)
                print(f"   ❌ Day {day:03d} ({date_str}): Error - {e}")
        
        # 生成总结
        elapsed = time.time() - start_time
        success_rate = success_count / total_days * 100
        
        print(f"\n📊 {year} Generation Summary:")
        print(f"   - Total days: {total_days}")
        print(f"   - Successful: {success_count}")
        print(f"   - Success rate: {success_rate:.1f}%")
        print(f"   - Time elapsed: {elapsed/60:.1f} minutes")
        print(f"   - Average rate: {success_count/(elapsed/60):.1f} files/minute")
        
        return success_count > 0
    
    def generate_multiple_years(self, years: list, resume: bool = True) -> bool:
        """生成多年特征栈"""
        print(f"\n🚀 Generating NO2 feature stacks for years: {years}")
        print("=" * 60)
        
        total_success = 0
        total_days = 0
        
        for year in years:
            year_success = self.generate_year(year, resume)
            if year_success:
                total_success += 1
            
            # 计算总天数
            total_days += 366 if calendar.isleap(year) else 365
        
        print(f"\n🎉 Multi-year Generation Summary:")
        print(f"   - Years processed: {len(years)}")
        print(f"   - Successful years: {total_success}")
        print(f"   - Total days: {total_days}")
        print(f"   - Success rate: {total_success/len(years)*100:.1f}%")
        
        return total_success > 0
    
    def _get_completed_files(self, year: int) -> set:
        """获取已完成的文件列表"""
        completed = set()
        
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                file_year, date_str, status = parts[0], parts[1], parts[2]
                                if file_year == str(year) and status == 'success':
                                    completed.add(date_str)
            except Exception as e:
                print(f"   ⚠️ Error reading progress file: {e}")
        
        return completed
    
    def _update_progress(self, year: int, day: int, date_str: str, success: bool):
        """更新进度文件"""
        try:
            with open(self.progress_file, 'a') as f:
                status = 'success' if success else 'failed'
                f.write(f"{year},{date_str},{status},{datetime.now().isoformat()}\n")
        except Exception as e:
            print(f"   ⚠️ Error updating progress file: {e}")
    
    def check_progress(self, year: int = None) -> dict:
        """检查生成进度"""
        if not os.path.exists(self.progress_file):
            return {"error": "Progress file not found"}
        
        try:
            with open(self.progress_file, 'r') as f:
                lines = f.readlines()
            
            progress = {}
            for line in lines:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        file_year, date_str, status = parts[0], parts[1], parts[2]
                        if year is None or file_year == str(year):
                            if file_year not in progress:
                                progress[file_year] = {'success': 0, 'failed': 0}
                            progress[file_year][status] += 1
            
            return progress
        except Exception as e:
            return {"error": f"Error reading progress: {e}"}

def main():
    """主函数"""
    print("🚀 NO2 Batch Feature Stack Generator (Fixed Version, Colab)")
    print("=" * 60)
    
    # 初始化批量生成器
    generator = NO2BatchGenerator()
    
    # 检查当前进度
    print("\n📊 Checking current progress...")
    progress = generator.check_progress()
    if "error" not in progress:
        for year, stats in progress.items():
            total = stats['success'] + stats['failed']
            success_rate = stats['success'] / total * 100 if total > 0 else 0
            print(f"   {year}: {stats['success']}/{total} ({success_rate:.1f}%)")
    
    # 生成年份配置
    years = [2019, 2020, 2021, 2022, 2023]
    
    print(f"\n🎯 Target years: {years}")
    print("💡 This will generate NO2 feature stacks for all years with resume capability")
    
    # 确认开始
    response = input("\n❓ Do you want to start batch generation? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🚀 Starting batch generation...")
        success = generator.generate_multiple_years(years, resume=True)
        
        if success:
            print("\n🎉 Batch generation completed successfully!")
            print("📋 Next steps:")
            print("   1. ✅ Feature stacks generated")
            print("   2. 🔄 Ready for NO2 scaler regeneration")
            print("   3. 🔄 Ready for model training")
        else:
            print("\n⚠️ Batch generation had issues")
    else:
        print("\n⏸️ Batch generation cancelled")

if __name__ == "__main__":
    main()
```

### Key Features

1. **Batch Processing**: Handles 2019-2023 data generation
2. **Resume Capability**: Can continue from where it left off
3. **Progress Tracking**: Saves progress to `batch_progress.txt`
4. **Performance Optimization**: Loads static features once
5. **Error Handling**: Comprehensive exception handling
6. **User Interface**: Interactive confirmation and progress display

### Dependencies

- `NO2IndependentFeatureStackBuilder`: Core feature stack builder class
- NumPy: For array operations and compressed saving
- Standard libraries: datetime, calendar, time, os, sys

### Output

- **Format**: `.npz` compressed NumPy arrays
- **Naming**: `NO2_stack_YYYYMMDD.npz`
- **Size**: ~4.6-5.1 MB per file
- **Organization**: By year in separate folders

### Usage

```bash
python batch_generate_no2_fixed_colab.py
```

This script is designed for Google Colab environment and handles the complete feature engineering pipeline for NO2 data.

---

## SO2 Feature Stack Generation Code

### Overview
This document also contains the implementation code for generating SO2 feature stacks, which includes additional SO2-specific enhancements for handling winter missingness and climatology features.

### SO2 Feature Stack Generation Script

**File**: `feature_stack_improved_so2.py`

```python
#!/usr/bin/env python3
"""
SO2特征栈生成脚本（改进版，Colab版本）
SO2 Feature Stack Generation (Improved Version, Colab Version)

使用改进后的代码生成SO2特征栈，包含冬季缺失值增强和气候学特征
"""

import rasterio
import xarray as xr
import numpy as np
import os
import pandas as pd
import re
from datetime import datetime, timedelta
from scipy.ndimage import generic_filter
import warnings
import calendar
import geopandas as gpd
from rasterio.features import rasterize

# Default to built-in PATHS. Optionally allow overriding via config when env USE_CONFIG_PATHS=1
BASE = "/content/drive/MyDrive"
PATHS = {
    # Target observations
    "SO2_DAILY": os.path.join(BASE, "GEE_SO2", "SO2_Daily_Multiband_{year}.tif"),
    # Precomputed parameters
    "SO2_LAG1": os.path.join(BASE, "Variables/so2_lag_1day", "SO2_Lag1day_CAMS_2019_2023.nc"),
    "SO2_NEI": os.path.join(BASE, "Variables", "SO2_Neighbor_Mean_2019_2023.tif"),
    "SO2_CLIMATOLOGY": os.path.join(BASE, "Variables/so2_monthly_climatology", "so2_monthly_climatology_2019_2023.tif"),
    # Day-of-year
    "SIN_DOY": os.path.join(BASE, "Variables/day_of_year", "sin_doy_{year}.tif"),
    "COS_DOY": os.path.join(BASE, "Variables/day_of_year", "cos_doy_{year}.tif"),
    # Weekday weights (SO2 specific if available)
    "WEIGHTS_SO2": os.path.join(BASE, "Variables", "so2_weekday_weight_2019_2023.csv"),
    # Static
    "DEM": os.path.join(BASE, "Variables/dem_aligned_s5p/dem_s5p_aligned.tif"),
    "SLOPE": os.path.join(BASE, "Variables/slope_aligned_s5p/slope_s5p_aligned.tif"),
    "POP": os.path.join(BASE, "Variables/population_aligned_s5p/population_2020_s5p_aligned_fixed.tif"),
    "LULC_DIR": os.path.join(BASE, "Variables/lulc_esa_onehot_s5p"),
    # Meteorology (daily multiband)
    "U10": os.path.join(BASE, "Variables/10m_u_component_of_wind/u10_processed/u10_daily_{year}.tif"),
    "V10": os.path.join(BASE, "Variables/10m_v_component_of_wind/v10_processed/v10_daily_{year}.tif"),
    "BLH": os.path.join(BASE, "Variables/boundary_layer_height/boundary_layer_height_processed/blh_daily_{year}.tif"),
    "TP": os.path.join(BASE, "Variables/total_precipitation/total_precipitation_processed/precipitation_daily_{year}.tif"),
    "T2M": os.path.join(BASE, "Variables/2m_temperature/2m_temperature_processed/temperature_daily_{year}.tif"),
    "SP": os.path.join(BASE, "Variables/surface_pressure/surface_pressure_processed/pressure_daily_{year}.tif"),
    "STR": os.path.join(BASE, "Variables/surface_net_thermal_radiation/thermal_radiation_processed/thermal_radiation_daily_{year}.tif"),
    "SSR_CLR": os.path.join(BASE, "Variables/surface_net_solar_radiation_clearsky/solar_radiation_processed/solar_radiation_daily_{year}.tif"),
    # Means for fallbacks
    "MEAN_U10": os.path.join(BASE, "Variables/10m_u_component_of_wind/u10_processed/u10_5year_mean.tif"),
    "MEAN_V10": os.path.join(BASE, "Variables/10m_v_component_of_wind/v10_processed/v10_5year_mean.tif"),
    # AOI (optional)
    "AOI": os.path.join(BASE, "AOI", "delimitazione_distretto.shp"),
}

# [Additional utility functions and main implementation code would be included here]
# [The full implementation includes 30 features with SO2-specific enhancements]

def create_single_day_feature_stack_so2(date_str, year, day_of_year, df_weights):
    """Create SO2 feature stack for a single day with winter missingness enhancements"""
    
    # S5P grid dimensions
    target_height, target_width = 300, 621
    
    # Initialize feature stack
    features = {}
    
    # [Complete implementation with 30 features including:]
    # 1. Static variables (DEM, slope, population)
    # 2. LULC one-hot encoding (10 classes)
    # 3. Meteorological variables (8 variables)
    # 4. Wind derivatives (wind speed, wind direction sin/cos)
    # 5. SO2 target variable
    # 6. SO2 lag1 with robust imputation
    # 7. SO2 neighbor mean
    # 8. SO2 monthly climatology (SO2-specific enhancement)
    # 9. Time features (sin/cos day of year)
    # 10. Weekday weights
    
    # Feature order (30 features total)
    feature_order = [
        'dem', 'slope', 'population',
        'lulc_class_10', 'lulc_class_20', 'lulc_class_30', 'lulc_class_40', 'lulc_class_50',
        'lulc_class_60', 'lulc_class_70', 'lulc_class_80', 'lulc_class_90', 'lulc_class_100',
        'u10', 'v10', 'ws', 'wd_sin', 'wd_cos', 'blh', 'tp', 't2m', 'sp', 'str', 'ssr_clear',
        'so2_lag1', 'so2_neighbor', 'so2_climate_prior',  # SO2-specific features
        'sin_doy', 'cos_doy', 'weekday_weight'
    ]
    
    # [Complete implementation continues...]
    
    return X, y, mask, feature_order, cont_idx, onehot_idx, noscale_idx, coverage, trainable, season, doy, weekday, year_len, lag1_fill_ratio, neighbor_fill_ratio

def save_so2_stack(date_str: str, out_base: str = "/content/drive/MyDrive/Feature_Stacks") -> str:
    """Build and save a single-day SO2 feature stack to NO2-aligned structure"""
    # Implementation for saving SO2 stacks in NO2-aligned directory structure
    
def generate_so2_year(year: int, out_base: str = "/content/drive/MyDrive/Feature_Stacks", resume: bool = True) -> None:
    """Generate and save a full-year SO2 stacks to NO2-aligned structure"""
    # Implementation for generating full year of SO2 feature stacks
```

### SO2-Specific Enhancements

1. **Winter Missingness Handling**: Enhanced imputation strategies for SO2's high winter missing rates (93.1%)

2. **Monthly Climatology**: SO2-specific climate prior feature (`so2_climate_prior`) to handle seasonal patterns

3. **Robust Lag1 Imputation**: Improved SO2 lag1 imputation with median filtering and fallback strategies

4. **SO2-Specific Weights**: Separate weekday weight tables for SO2 vs NO2 patterns

5. **Lower Coverage Threshold**: Uses 1% coverage threshold (vs higher for NO2) due to SO2's higher missing rates

### Key Differences from NO2 Implementation

- **30 features** (vs 27 for NO2) - includes SO2 climatology
- **Enhanced imputation** for high missing rate scenarios
- **SO2-specific file paths** and variable names
- **Climate prior integration** for seasonal patterns
- **Robust error handling** for winter data gaps

### Output Structure

- **Format**: `.npz` compressed NumPy arrays
- **Naming**: `SO2_stack_YYYYMMDD.npz`
- **Organization**: `/Feature_Stacks/SO2_{year}/` (aligned with NO2 structure)
- **Features**: 30 features including SO2-specific enhancements

### Usage

```python
# Test single day
test_single_day_so2()

# Generate full year
generate_so2_year(2019)

# Save single day
save_so2_stack("2019-09-01")
```

This SO2 implementation addresses the unique challenges of SO2 data, particularly the high winter missing rates and seasonal patterns, while maintaining compatibility with the NO2 feature stack structure.
