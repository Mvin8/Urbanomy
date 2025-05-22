Urbanomy: Investment Attractiveness Analysis for Land-Use
=========================================================
.. logo-start

.. figure:: https://sun9-46.userapi.com/impf/aUFBStH0x_6jN9UhgwrKN1WN4hZ9Y2HMMrXT2w/NuzVobaGlZ0.jpg?size=1590x400&quality=95&crop=0,0,1878,472&sign=9d33baa41a86de35d951d4bbd8011994&type=cover_group
   :alt: The Institute of Design and Urban Studies

.. figure:: https://psv4.userapi.com/s/v1/d/LQL8fQ5o0kL1VRqeuM_P9hNUklLHJ6C8muuas7YDuhKolUDx7ZO9Fw1IszJZQNHfnNqXTnzTAz5E8R2WNythf40XWGq5RO7GIVbBiSOn2CLhZravVGB-bw/ChatGPT_Image_22_maya_2025_g__18_15_35.png
   :alt: Новый логотип Urbanomy

.. logo-end

.. logo-end

|Documentation Status| |PythonVersion| |Black|

.. readme-start 

Overview
--------
Urbanomy is a Python library for computing **spatial** and **economic** investment-attractiveness metrics
for land-use polygons. It integrates GeoPandas workflows with financial models (NPV, IRR, ROI, payback
period, economic index) and produces both tabular summaries and geospatial visualizations.

Features
--------
- Flexible per-land-use benchmark definitions and weighting schemes  
- Automated cash-flow generation and aggregation  
- Calculation of NPV, IRR, ROI, payback period, economic index (EI)  
- Spatial–economic synthesis into a single INV score  
- Utilities for normalization, quantization, and data cleaning  
- Plotting routines for attribute weights and map grids  

Installation
------------
Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/vasilstar97/urbanomy.git@main

Quickstart
----------
Load your GeoDataFrame, run the analyzer, and inspect results:

.. code-block:: python

   import geopandas as gpd
   from urbanomy.investment_attractiveness import InvestmentAttractivenessAnalyzer

   # 1. Read your land-use GeoJSON
   gdf = gpd.read_file("examples/data/landuse_sample.geojson")

   # 2. Define benchmarks (see docs for schema)
   benchmarks = {
       "residential_individual": { "density": 1.0, "cost_build": 1000, ... },
       "industrial":            { "density": 0.5, "cost_build": 800,  ... },
       # ...
   }

   # 3. Compute metrics
   analyzer = InvestmentAttractivenessAnalyzer(benchmarks=benchmarks)
   enriched_gdf, summary = analyzer.calculate_investment_metrics(gdf)

   # 4. View summary
   print(summary)

Data
----
Sample data and templates are provided in the `examples/data/` directory.
Feel free to replace with your own GeoJSON, Shapefile or PostGIS source.

Examples
--------
Complete usage examples are available as Jupyter notebooks:

1. `examples/landuse_analysis.ipynb`  
2. `examples/voronoi_zones.ipynb`  

Documentation
-------------
Full documentation, including API reference and tutorials, is hosted on Read the Docs:

https://urbanomy.readthedocs.io/

Contributing
------------
Contributions are very welcome! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/your-feature`)  
3. Commit your changes with clear messages  
4. Push to your fork and open a Pull Request  

Please follow Black formatting and add tests for new functionality.

License
-------
This project is licensed under the **BSD-3-Clause** License. See `LICENSE` for details.

Acknowledgments
---------------
- Developed at the Institute of Design and Urban Studies  
- Inspired by best practices in geospatial finance and urban analytics  

Contact
-------
Maxim Natykin  
Telegram: https://t.me/Mvin98  
GitHub Issues: https://github.com/vasilstar97/urbanomy/issues  

Publications
------------
- See also related work at https://scholar.google.com/