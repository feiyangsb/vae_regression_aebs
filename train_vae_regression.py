"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-03-15 22:16:46
@modify date 2020-03-15 22:16:46
@desc [description]
"""

from scripts.vae_regression_trainer import VAERegression

data_path = "../robust_vae/data/training/"
trainer = VAERegression(data_path, 350)
trainer.fit()
trainer.save_model()