import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict as edict
from rich.console import Console
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from ML_trainer.build_model import build_model
from configs import TrainConfig
from data.build_data import load_data
from utils_.metrics import return_result, return_result_by_threshold
from utils_.utils import set_seed, get_scaler

warnings.filterwarnings(action='ignore')


def main():

    config = TrainConfig.parse_arguments()
    console = Console(color_system='256', force_terminal=True, width=160)
    console.log(config.__dict__)
    config.save()

    set_seed(config.random_state)

    # load data
    X, y = load_data(data_dir=config.data_dir,
                     file_name=config.file_name,
                     target=config.target,
                     verbose=config.verbose)

    # train/test split samples
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=(1-config.train_ratio),
                                                        shuffle=False,
                                                        random_state=config.random_state)

    # scaling data
    if config.scaler:
        scaler = get_scaler(config.scaler)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    model_params = build_model(random_state=config.random_state)

    if config.model_names[0] == "all":
        model_list = list(model_params.keys())
    else:
        model_list = config.model_names

    # save results
    result_df = pd.DataFrame()
    result_thres_df = pd.DataFrame()

    # Training!
    with console.status(f"[bold green] Working on tasks...") as status:
        for model_name in model_list:
            model, model_param = model_params[model_name]['model'], model_params[model_name]['params']

            gcv = GridSearchCV(estimator=model, param_grid=model_param, n_jobs=3)
            gcv.fit(X_train, y_train)

            ## Total result
            train_result = return_result(gcv.best_estimator_, X_train, y_train)
            test_result = return_result(gcv.best_estimator_, X_test, y_test)

            result = edict()
            result.random_state = config.random_state
            result.model_name = model_name

            for key in test_result.keys():
                result[f'train_{key}'] = train_result[key]
                result[f'test_{key}'] = test_result[key]

            result = pd.DataFrame.from_dict([result])
            result_df = pd.concat([result_df, result])

            ## Threshold result
            thres_list = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

            for (min_v, max_v) in thres_list:
                result_threshold = edict()
                result_threshold.random_state = config.random_state
                result_threshold.model_name = model_name

                result_threshold.min_v = min_v
                result_threshold.max_v = max_v

                train_result_threshold = return_result_by_threshold(gcv.best_estimator_, X_train, y_train, min_v, max_v)
                test_result_threshold = return_result_by_threshold(gcv.best_estimator_, X_test, y_test, min_v, max_v)

                for key in test_result_threshold.keys():
                    result_threshold[f'train_{key}'] = train_result_threshold[key]
                    result_threshold[f'test_{key}'] = test_result_threshold[key]

                result_threshold = pd.DataFrame.from_dict([result_threshold])
                result_thres_df = pd.concat([result_thres_df, result_threshold])


            # save result_df
            result_df.to_csv(os.path.join(config.checkpoint_dir, "result.csv"), index=False)
            result_thres_df.to_csv(os.path.join(config.checkpoint_dir, "result_thres.csv"), index=False)

            # confusion matrix
            if config.confusion_matrix:
                label = sorted(y_test.unique().tolist())

                plot = plot_confusion_matrix(gcv.best_estimator_,
                                             X_test,
                                             y_test,
                                             display_labels=label,
                                             cmap=plt.cm.Blues,
                                             normalize=None)

                plt.savefig(os.path.join(config.checkpoint_dir, 'plot', f"{model_name}_confusion_matrix.png"), dpi=350)


            if config.curve_plot:
                plot = plot_roc_curve(gcv.best_estimator_, X_test, y_test)
                plt.savefig(os.path.join(config.checkpoint_dir, 'plot', f'{model_name}_roc_curve.png'), dpi=350)


            console.log(f"{model_name} model complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()