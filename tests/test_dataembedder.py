import os
import unittest
#from opencmiss.utils.zinc.field import createFieldMeshIntegral
#from opencmiss.zinc.field import Field
#from opencmiss.zinc.node import Node, Nodeset
#from opencmiss.zinc.result import RESULT_OK
from dataembedder.dataembedder import DataEmbedder

here = os.path.abspath(os.path.dirname(__file__))

def assertAlmostEqualList(testcase, actualList, expectedList, delta):
    assert len(actualList) == len(expectedList)
    for actual, expected in zip(actualList, expectedList):
        testcase.assertAlmostEqual(actual, expected, delta=delta)

class DataEmbedderTestCase(unittest.TestCase):

    def test_embed_cube_square_line(self):
        """
        Test embedding an example model consisting of a cube, square and line elements embedded in two cubes mesh.
        """

        zincScaffoldFileName = os.path.join(here, "resources", "body_two_cubes_scaffold.exf")
        zincFittedGeometryFileName = os.path.join(here, "resources", "body_two_cubes_fitted0.exf")
        zincDataFileName = os.path.join(here, "resources", "data_cube_square_line.exf")
        dataEmbedder = DataEmbedder(zincScaffoldFileName, zincFittedGeometryFileName, zincDataFileName)
        dataEmbedder.load()
        # check fields and group data automatically determined
        self.assertEqual("coordinates", dataEmbedder.getDataCoordinatesField().getName())
        self.assertEqual("fitted coordinates", dataEmbedder.getFittedCoordinatesField().getName())
        self.assertEqual("body coordinates", dataEmbedder.getMaterialCoordinatesField().getName())
        self.assertEqual("marker", dataEmbedder.getDataMarkerGroup().getName())
        groupNames = dataEmbedder.getGroupNames()
        self.assertTrue(dataEmbedder.groupIsEmbed("cube"))
        self.assertTrue(dataEmbedder.groupIsEmbed("square"))
        self.assertTrue(dataEmbedder.groupIsEmbed("line"))
        self.assertTrue(dataEmbedder.groupIsEmbed("ICN"))
        self.assertFalse(dataEmbedder.groupIsEmbed("bottom"))
        self.assertFalse(dataEmbedder.groupIsEmbed("marker"))
        self.assertFalse(dataEmbedder.groupIsEmbed("tip 2"))
        self.assertFalse(dataEmbedder.groupExists("tip 1"))

if __name__ == "__main__":
    unittest.main()
