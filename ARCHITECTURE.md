# 🏗️ Sharks from Space - System Architecture

## Overview
The Sharks from Space system is a modular Python backend that processes NASA satellite data to generate shark foraging hotspot predictions. The system follows a pipeline architecture with clear separation of concerns.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sharks from Space System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   PACE      │    │ MODIS-Aqua  │    │    SWOT     │         │
│  │ Ocean Color │    │ SST/Chl/Kd  │    │    SSH      │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│           │                 │                 │                │
│           └─────────────────┼─────────────────┘                │
│                             │                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Data Loading & Harmonization                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │data_loader  │  │ harmonize   │  │validation  │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Derived Fields Computation                     ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │Geostrophic  │  │EKE/Vorticity│  │SST Gradient│         ││
│  │  │Velocities   │  │             │  │Euphotic Dep│         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Normalization & Model Core                     ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │Robust 5-95  │  │FOI Logistic │  │Model       │         ││
│  │  │Percentile   │  │Regression   │  │Validation  │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Output Generation                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │NetCDF/GeoTIFF│  │Interactive │  │Static      │         ││
│  │  │Files        │  │Dashboard   │  │Snapshots   │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Ingestion
- **PACE**: Ocean color data (chlorophyll-a, plankton type)
- **MODIS-Aqua**: SST, chlorophyll-a, Kd490 (8-day composites)
- **SWOT**: Sea surface height (21-day cycle)

### 2. Temporal Alignment
- **Time Window**: 2025-03-01 to 2025-03-14 (14 days)
- **Composite Strategy**: 7-day rolling mean
- **Synchronization**: All datasets aligned to common time grid

### 3. Spatial Harmonization
- **Target Resolution**: 0.1° × 0.1° grid
- **Regridding**: Linear interpolation to common grid
- **Region**: Coral Triangle (94°E-142°E, 11°S-6°N)

### 4. Derived Field Computation
- **Geostrophic Velocities**: u_g = -(g/f)∂SSH/∂y, v_g = (g/f)∂SSH/∂x
- **Eddy Kinetic Energy**: EKE = 0.5·(u_g'² + v_g'²)
- **Relative Vorticity**: ζ = ∂v_g/∂x - ∂u_g/∂y
- **SST Gradient**: |∇SST| = √((∂SST/∂x)² + (∂SST/∂y)²)
- **Euphotic Depth**: Z_eu = 4.6/Kd490

### 5. Normalization
- **Method**: Robust 5-95 percentile scaling
- **Formula**: X~ = clip((X-P5)/(P95-P5), 0, 1)
- **Purpose**: Standardize variables for model input

### 6. FOI Model
- **Thermal Suitability**: ST = exp(-(T-T_pref)²/(2σ_T²))
- **Eddy Relief**: ST,eff = 1 - (1-ST)(1-α·EKE~)
- **Productivity**: P = log(1+Chl)
- **Twilight Access**: A_tw = β₁·Z~_eu + β₂·EKE~
- **Front Strength**: F_front = |∇SST|~
- **Final FOI**: η = b₀ + b₁·ST,eff + b₂·P~ + b₃·A_tw + b₄·F_front
- **Logistic**: FOI = 1/(1 + e^(-η))

### 7. Visualization
- **Basemap**: NASA GIBS MODIS True Color
- **Overlay**: Semi-transparent FOI raster
- **Interactive**: Folium-based web interface
- **Static**: PNG snapshots for presentations

## Module Dependencies

```
predictor.py (Main Orchestrator)
├── data_loader.py (Satellite Data Ingestion)
├── harmonize.py (Temporal/Spatial Alignment)
├── derived_fields.py (Oceanographic Analysis)
├── normalization.py (Data Standardization)
├── model_core.py (FOI Mathematical Model)
└── visualize_hotspots.py (Interactive Visualization)
```

## Configuration Management

- **config.yaml**: Central configuration file
- **Parameters**: Temporal windows, spatial bounds, model coefficients
- **Flexibility**: Easy modification for different regions/time periods
- **Validation**: Input validation and error handling

## Output Structure

```
data/outputs/
├── foi/YYYYMMDD/
│   ├── foi_map.nc          # NetCDF FOI data
│   ├── foi_summary.json    # Statistical summary
│   └── metadata.json       # Processing metadata
├── cps/YYYYMMDD/           # Conservation Priority Surface
└── visualizations/
    ├── foi_dashboard.html  # Interactive map
    ├── foi_snapshot.png    # Static image
    └── foi_statistics.png  # Statistical plots
```

## Error Handling

- **Graceful Degradation**: System continues with available data
- **Comprehensive Logging**: Detailed logs for debugging
- **Validation**: Data quality checks at each step
- **Recovery**: Automatic retry mechanisms for transient failures

## Performance Considerations

- **Memory Management**: Efficient xarray operations
- **Parallel Processing**: Multi-threaded data processing
- **Caching**: Intermediate results stored for reuse
- **Optimization**: Vectorized operations where possible

## Extensibility

- **Modular Design**: Easy to add new data sources
- **Plugin Architecture**: Custom derived fields
- **API Ready**: FastAPI endpoints for web integration
- **Machine Learning**: Ready for ML model integration

## Security & Reliability

- **Input Validation**: All inputs validated
- **Error Boundaries**: Isolated failure domains
- **Data Integrity**: Checksums and validation
- **Audit Trail**: Complete processing logs
