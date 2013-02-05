from base.archiver import Archiver
import numpy

class FileArchiver(Archiver):
    """docstring for FileArchiver"""
    def __init__(self):
        super(FileArchiver, self).__init__()

    def save_results(self, target_data, prediction, filename='results.csv'):
        """Output format example:
        prediction,target,feature,note1,note2...
        3.00e0,3.10e0,5.00e0,proteinX,ab
        1.50e0,2.70e0,9.00e0,proteinY,b
        4.20e0,4.20e0,3.00e0,proteinZ,a
        """
        target_df = target_data.to_data_frame()
        prediction_array = prediction.as_array()
        error_msg = "%s %s" % (len(prediction_array), len(target_df))
        assert len(prediction_array) == len(target_df), error_msg
        target_df['prediction'] = prediction_array
        target_df.to_csv(filename)


class CoopCollectionFileArchiver(Archiver):
    """docstring for CoopCollectionFileArchiver"""
    def __init__(self):
        super(CoopCollectionFileArchiver, self).__init__()

    def save_results(self, target_data, prediction, filename='results.csv'):
        """Output format:
        prediction,target,feature,note1,note2...
        Example:
        3.00e0,3.10e0,5.00e0,proteinX,ab
        1.50e0,2.70e0,9.00e0,proteinY,b
        4.20e0,4.20e0,3.00e0,proteinZ,a
        """
        target_df = target_data.to_data_frame()
        feature = target_data.get_feature()
        prediction_array = prediction.as_array_from_id_list(feature)
        error_msg = "%s %s" % (len(prediction_array), len(target_df))
        assert len(prediction_array) == len(target_df), error_msg
        target_df['prediction'] = prediction_array
        target_df.to_csv(filename)


class TemperatureDependenceFileArchiver(Archiver):
    """docstring for TemperatureDependenceFileArchiver"""
    def __init__(self):
        super(TemperatureDependenceFileArchiver, self).__init__()

    def save_results(self, fold_target_data=None, fold_prediction=None,
                     unfold_target_data=None, unfold_prediction=None,
                     filename='results.csv'):
        if fold_target_data and fold_prediction:
            target_df = fold_target_data.to_data_frame()
            prediction_array = fold_prediction.as_array()
            error_msg = "%s %s" % (len(prediction_array), len(target_df))
            assert len(prediction_array) == len(target_df), error_msg
            target_df['prediction'] = prediction_array
            fold_filename = "fold_%s" % filename
            target_df.to_csv(fold_filename)
            print "Wrote", fold_filename
        else:
            pass

        if unfold_target_data and unfold_prediction:
            target_df = unfold_target_data.to_data_frame()
            prediction_array = unfold_prediction.as_array()
            error_msg = "%s %s" % (len(prediction_array), len(target_df))
            assert len(prediction_array) == len(target_df), error_msg
            target_df['prediction'] = prediction_array
            unfold_filename = "unfold_%s" % filename
            target_df.to_csv(unfold_filename)
            print "Wrote", unfold_filename
        else:
            pass
