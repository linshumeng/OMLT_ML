import lightgbm_model

spts, lvs, th = lightgbm_model.parse_model(r'C:\Users\yufengq\Desktop\RA\OMLT_ML\linear.txt')

import pprint
pp = pprint.PrettyPrinter(indent = 4)
pp.pprint(spts)
pp.pprint(lvs)
pp.pprint(th)
