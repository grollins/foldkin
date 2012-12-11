import base.archiver
import numpy

class FileArchiver(base.archiver.Archiver):
    """docstring for FileArchiver"""
    def __init__(self):
        super(FileArchiver, self).__init__()

    def save_results(self, target_data, prediction, filename='results.txt'):
        """Output format:
        prediction,target,feature,note1,note2...
        Example:
        3.00e0,3.10e0,5.00e0,proteinX,ab
        1.50e0,2.70e0,9.00e0,proteinY,b
        4.20e0,4.20e0,3.00e0,proteinZ,a
        """
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        notes = target_data.get_notes()
        prediction_array = prediction.as_array()
        assert len(feature_array) == len(target_array)
        assert len(prediction_array) == len(target_array)
        for this_note in notes:
            assert len(this_note) == len(target_array)
        f = open(filename, 'w')
        for i in xrange(len(target_array)):
            f.write("%.2e,%.2e,%.2e" % (prediction_array[i], target_array[i],
                                        feature_array[i]))
            for this_note in notes:
                f.write(",%s" % this_note[i])
            f.write("\n")
        f.close()
