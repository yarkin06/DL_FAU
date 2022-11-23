import unittest
import numpy as np
import tabulate
import argparse

ID = 0  # identifier for dispatcher

class TestCheckers(unittest.TestCase):

    def setUp(self):
        # Loads the reference images

        self.reference_img = np.load('reference_arrays/checker.npy')
        self.reference_img2 = np.load('reference_arrays/checker2.npy')

    def testPattern(self):
        # Creates a checkerboard pattern with resolution 250x250
        # and a tile_size of 25 and compares it to the reference image

        import pattern
        c = pattern.Checker(250, 25)
        c.draw()
        np.testing.assert_almost_equal(c.output, self.reference_img)

    def testPatternDifferentSize(self):
        # Creates a checkerboard pattern with resolution 100x100
        # and a tile_size of 25 and compares it to the reference image

        import pattern
        c = pattern.Checker(100, 25)
        c.draw()
        np.testing.assert_almost_equal(c.output, self.reference_img2)

    def testReturnCopy(self):
        # Checks whether the output of the pattern is a copy of the output object rather than the output object itself.

        import pattern
        c = pattern.Checker(100, 25)
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 res, c.output, "draw() did not return a copy!")


class TestCircle(unittest.TestCase):
    def setUp(self):
        # Loads the reference images

        self.reference_img = np.load('reference_arrays/circle.npy')
        self.reference_img2 = np.load('reference_arrays/circle2.npy')

    def _IoU(self,array1,array2):
        # Utility function returning the intersection over union value
        intersection = np.sum(array1*array2)
        union = array1+array2
        # union[union==2] = 1
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union
        return iou

    def testPattern(self):
        # Creates an image of a circle with resolution 1024x1024 a radius of 200 with a center at
        # (512,256) and compares it to the reference image using the IoU metric

        import pattern
        c = pattern.Circle(1024, 200, (512, 256))
        circ = c.draw()
        iou = self._IoU(circ,self.reference_img)
        self.assertAlmostEqual(iou,1.0,2)

    def testPatternDifferentSize(self):
        # Creates an image of a circle with resolution 512x512 a radius of 20 with a center at
        # (50,50) and compares it to the reference image using the IoU metric
        import pattern
        c = pattern.Circle(512, 20, (50, 50))
        circ = c.draw()
        iou = self._IoU(circ, self.reference_img2)
        self.assertAlmostEqual(iou, 1.0, 1)


    def testReturnCopy(self):
        # Checks whether the output of the pattern is a copy of the output object rather than the output object itself.
        import pattern
        c = pattern.Circle(512, 20, (50, 50))
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 res, c.output, "draw() did not return a copy!")


# Skipping the Spectrum  tests, if Spectrum is not implemented
#SPECTRUM_TEST = None
#try:
#    import pattern
#    pattern.Spectrum # checking if Spectrum exists
#    SPECTRUM_TEST = True
#except AttributeError:
#    SPECTRUM_TEST = False

# @unittest.skipIf(NO_SPECTRUM, "Spectrum is optional") # making it optional - but might confuse one or another
#if SPECTRUM_TEST:
class TestSpectrum(unittest.TestCase):

   
    def setUp(self):
        # Loads the reference images
        self.reference_img = np.load('reference_arrays/spectrum.npy')
        self.reference_img2 = np.load('reference_arrays/spectrum2.npy')


    def testPattern(self):
        # Creates an RGB spectrum with resolution 255x255x3 and compares it to the reference image
        import pattern
        s = pattern.Spectrum(255)
        spec = s.draw()
        np.testing.assert_almost_equal(spec, self.reference_img, decimal=2)

    def testPatternDifferentSize(self):
        # Creates an RGB spectrum with resolution 100x100x3 and compares it to the reference image
        import pattern
        s = pattern.Spectrum(100)
        spec = s.draw()
        np.testing.assert_almost_equal(spec, self.reference_img2, decimal=2)

    def testReturnCopy(self):
        # Checks whether the output of the pattern is a copy of the
        # output object rather than the output object itself.
        import pattern
        c = pattern.Spectrum(100)
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 res, c.output, "draw() did not return a copy!")


class TestGen(unittest.TestCase):
    def setUp(self):
        # Set the label and the file path
        self.label_path = './Labels.json'
        self.file_path = './exercise_data/'

    def _get_corner_points(self, image):
        # Utility function to check whether the augmentations where performed
        # expects batch of image - expected shape is [s,x,y,c]
        return image[:, [0, -1], :, :][:, : , [0, -1], :]

    def testInit(self):
        # Creates two image generator objects without shuffling.
        # Calling next on either one should result in the same output
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        gen2 = ImageGenerator(self.file_path, self.label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        np.testing.assert_almost_equal(gen.next()[0], gen2.next()[0])
        np.testing.assert_almost_equal(gen.next()[1], gen2.next()[1])

    def testDuplicate(self):
        # Image Generator without overlapping batches.
        # Choose one output sample at random and check if it is the only instance in both batches.
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 50, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        b1 = gen.next()
        b2 = gen.next()
        sample_index = np.random.choice(np.arange(50))
        sample = b1[0][sample_index]
        b1_without_sample = np.delete(b1[0], sample_index, axis=0)
        self.assertFalse(np.any(np.all(sample == b1_without_sample, axis=(1, 2, 3))))
        self.assertFalse(np.any(np.all(sample == b2[0], axis=(1, 2, 3))))


    def testResetIndex(self):
        # Data contains 100 image samples, for a batchsize of example 60 an
        # overlap of 20 occurs, therefore the first 20 elements
        # of the first batch should be equal to the last 20 of the second batch
        # check with two different batch sizes
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 60, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        gen2 = ImageGenerator(self.file_path, self.label_path, 83, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        b1 = gen.next()[0]
        b2 = gen.next()[0]
        np.testing.assert_almost_equal(b1[:20], b2[40:])
        b1 = gen2.next()[0]
        b2 = gen2.next()[0]
        np.testing.assert_almost_equal(b1[:66], b2[17:])
        self.assertFalse(b1[65] is b2[-1])  # Check if it is a shared object


    def testShuffle(self):
        # Creates two image generator objects.
        # Since shuffle is enabled for one image generator the output should be different.
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=True)
        gen2 = ImageGenerator(self.file_path, self.label_path, 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, gen.next()[0], gen2.next()[0])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, gen.next()[1], gen2.next()[1])

    def testShuffleEpoch(self):
        # Checks for shuffling of complete data set, not only inside batch after one epoch
        # Detects if the same images (or rather ixel values) are included in the same batch
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 50, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=True)
        b1_epoch0 = gen.next()[0]
        gen.next()
        b1_epoch1 = gen.next()[0]
        self.assertFalse(np.all(np.sort(b1_epoch0, axis=None) == np.sort(b1_epoch1, axis=None)))

    def testEpoch(self):
        from generator import ImageGenerator
        gen = ImageGenerator(self.file_path, self.label_path, 50, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=True)
        gen.next()
        self.assertEqual(0, gen.current_epoch())
        gen.next()
        self.assertEqual(0, gen.current_epoch())
        gen.next()
        self.assertEqual(1, gen.current_epoch())

    def testRotation(self):
        from generator import ImageGenerator

        batch1 = ImageGenerator(self.file_path, self.label_path, 100, [32, 32, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        batch2 = ImageGenerator(self.file_path, self.label_path, 100, [32, 32, 3], rotation=True, mirroring=False, shuffle=False).next()[0]

        # determine the images which were augmented
        augmented_images_indices = np.sum(np.abs(batch1 - batch2), axis=(1, 2, 3)).astype(np.bool_)

        # extract corner points for each sample and reduce augmented to one characateristic row
        # this row is also inverted for simpler computations
        augmented_corners = self._get_corner_points(batch2[augmented_images_indices])
        characteristic_corners = augmented_corners[:, 0, ::-1]
        original_corners = self._get_corner_points(batch1[augmented_images_indices])

        # subtract characteristic corners to original corners. after summing spacial and channel dimensions, 0 are expected
        # e.g. if sample 2 is rotated by 90, we have a 0 in rot1 at the 2nd posistion see similiar to testMirroring
        rot1 = np.sum(original_corners[:, :, 0] - characteristic_corners, axis=(1, 2))
        rot2 = np.sum(original_corners[:, 1, :] - characteristic_corners, axis=(1, 2))
        rot3 = np.sum(original_corners[:, ::-1, 1] - characteristic_corners, axis=(1, 2))

        # assumption is that augmented images are either rotated by 90 (rot1), 180 (ro2) or 270 (rot3) degrees, thus
        # their elementwise product must be zero
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, batch1, batch2)
        np.testing.assert_almost_equal(np.sum(rot1 * rot2 * rot3), 0)

    def testMirroring(self):
        from generator import ImageGenerator

        batch1 = ImageGenerator(self.file_path, self.label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        batch2 = ImageGenerator(self.file_path, self.label_path, 12, [32, 32, 3], rotation=False, mirroring=True, shuffle=False).next()[0]

        # determine the images which were augmented
        augmented_images_indices = np.sum(np.abs(batch1 - batch2), axis=(1, 2, 3)).astype(np.bool_)

        # extract corner points for each sample
        augmented_corners = self._get_corner_points(batch2[augmented_images_indices])
        original_corners = self._get_corner_points(batch1[augmented_images_indices])

        # pick vertical ref points and compare them with opposites in original corners
        # compute then the diff and sum over vertical axis and channels. those images which were flipped vertically
        # contain now zeros
        vertical_augmented_corners = augmented_corners[:, :, 0]
        vertical = np.sum(original_corners[:, :, 1, :] - vertical_augmented_corners, axis=(1, 2))

        # pick horizontal ref points and compare them with opposites in original corners
        # compute then the diff and sum over horizontal axis and channels. those images which were flipped horizontally
        # contain now zeros
        horizontal_augmented_corners = augmented_corners[:, 0, :]
        horizontal = np.sum(original_corners[:, 1, :, :] - horizontal_augmented_corners, axis=(1, 2))

        # pick top left corner and bottom right corner (diagonals are flipped if double mirror)
        vertical_horizontal_augmented_corners = np.stack(
            list(zip(augmented_corners[:, 0, 0, :], augmented_corners[:, 1,  1, :])))
        original_corner_diagonals = np.stack(
            list(zip(original_corners[:, 1, 1, :], original_corners[:, 0, 0, :])))
        horizontal_vertical = np.sum(original_corner_diagonals - vertical_horizontal_augmented_corners, axis=(1, 2))


        # the elementwise product of horizontal and vertical must be zero
        # since the images can only be augmented with vertical or horizontal mirroring
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, batch1, batch2)
        np.testing.assert_almost_equal(np.sum(vertical * horizontal * horizontal_vertical), 0)

    def testResize(self):
        from generator import ImageGenerator

        batch = ImageGenerator(self.file_path, self.label_path, 12, [50, 50, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        self.assertEqual(batch.shape, (12, 50, 50, 3))

    def testLabels(self):
        # this test makes sure your generator returns integers as labels and not any strings or floating point values
        # if this one fails make sure you cast your array values to integers
        from generator import ImageGenerator
        label = ImageGenerator(self.file_path, self.label_path, 12, [50, 50, 3], rotation=False, mirroring=False,
                               shuffle=False).next()[1]

        self.assertFalse(isinstance(label[0], str))
        self.assertTrue(np.issubdtype(np.array(label).dtype, np.integer))

if __name__ == '__main__':

    import sys
    if sys.argv[-1] == "Bonus":
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestCheckers, TestCircle, TestSpectrum, TestGen]
        percentages = [10, 10, 10, 70]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 1
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p), "{:.3f} / 10 (%)".format(p/100 * exam_percentage)])
        table.append([])
        table.append(["Ex0", "Total Achieved", "", "{} / 100 (%)".format(total_points), "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])
        print(tabulate.tabulate(table,
                                headers=['Pos', 'Test', "Result", 'Percent in Exercise', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()
