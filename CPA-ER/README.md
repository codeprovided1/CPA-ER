# CPA-ER 

You can download the Amazon datasets from [here.](https://nijianmo.github.io/amazon/)

If you want to process your own data, you will need: user-item interaction data, user and item review information (for extracting auxiliary information), and relationship data related to items, such as brand, category, attributes, etc., that can be processed into knowledge graph triples.

You can use the `dataset_preprocessing.py` script in the `scripts` folder to preprocess your dataset.

Then, run `train_base_rec.py` to train the rating prediction model.

The detail of counterfactual reasoning model is located in the `cnt_model.py` file.

You can obtain the user's preferences by running `generate_cnt_pre.py`.

If u want try more experients about our model, you can change params you need in `train_base_rec.py` and `generate_cnt_pre.py`, and run them again.

Finally, you can integrate the learned path augmentation information into any reinforcement-based path reasoning model. You just need to modify the rewards in the reinforcement learning environment. For detailed instructions, please refer to the paper.

