import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Pandas settings
pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 60
pd.options.display.float_format = '{:,.4f}'.format

# Matplotlib settings
plt.style.use('default')
rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = '#cccccc'
rcParams['savefig.dpi'] = 300

# Colors
TRAIN_COLOR = "#6699cc"
TEST_COLOR = "#efcb68"
FORECAST_COLOR = "#8f2d56"