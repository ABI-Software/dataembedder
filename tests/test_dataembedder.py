import os
import unittest
from opencmiss.utils.zinc.field import get_group_list
from opencmiss.utils.zinc.finiteelement import evaluate_field_nodeset_range
from opencmiss.zinc.field import Field
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
        self.assertEqual(7, len(groupNames))
        self.assertTrue(dataEmbedder.isGroupEmbed("cube"))
        self.assertTrue(dataEmbedder.isGroupEmbed("square"))
        self.assertTrue(dataEmbedder.isGroupEmbed("line"))
        self.assertTrue(dataEmbedder.isGroupEmbed("ICN"))
        self.assertFalse(dataEmbedder.isGroupEmbed("bottom"))
        self.assertFalse(dataEmbedder.isGroupEmbed("marker"))
        self.assertFalse(dataEmbedder.isGroupEmbed("tip 2"))
        self.assertFalse(dataEmbedder.hasGroup("tip 1"))
        self.assertEqual(3, dataEmbedder.getGroupDimension("cube"))
        self.assertEqual(2, dataEmbedder.getGroupDimension("square"))
        self.assertEqual(1, dataEmbedder.getGroupDimension("line"))
        self.assertEqual(0, dataEmbedder.getGroupDimension("ICN"))
        self.assertEqual(1, dataEmbedder.getGroupSize("cube"))
        self.assertEqual(1, dataEmbedder.getGroupSize("square"))
        self.assertEqual(1, dataEmbedder.getGroupSize("line"))
        self.assertEqual(4, dataEmbedder.getGroupSize("ICN"))
        # test setting and unsetting embed flag
        dataEmbedder.setGroupEmbed("bottom", True)
        self.assertTrue(dataEmbedder.isGroupEmbed("bottom"))
        dataEmbedder.setGroupEmbed("bottom", False)
        self.assertFalse(dataEmbedder.isGroupEmbed("bottom"))

        outputRegion = dataEmbedder.generateOutput()
        outputRegion.writeFile("c:/Users/gchr006/oc/src/dataembedder/tests/resources/km_output.exf")
        # check output
        outputFieldmodule = outputRegion.getFieldmodule()
        outputBodyCoordinates = outputFieldmodule.findFieldByName("body coordinates").castFiniteElement()
        self.assertTrue(outputBodyCoordinates.isValid())
        outputGroups = get_group_list(outputFieldmodule)
        self.assertEqual(5, len(outputGroups))
        expectedBodyCoordinatesRange = {
            'ICN': ([0.4234094227546622, 0.48078031509540764, 0.24053424057686418],
                [0.4714586372862805, 0.5096098437995359, 0.2789736122606712]),
            'cube': ([0.13511413479016324, 0.21170471114299036, 0.11560628129407298],
                [0.7117047108476393, 0.7882952895792783, 0.6921968588706179]),
            'line': ([1.2882952842495057, 0.21170471154682882, 0.11560628082571832],
                [1.8648858589797928, 0.21170471264989535, 0.11560628106112401]),
            'marker': ([0.4234094227546622, 0.48078031509540764, 0.24053424057686418],
                [0.4714586372862805, 0.5096098437995359, 0.2789736122606712]),
            'square': ([0.7117047095185545, 0.21170471264989535, 0.11560628106112401],
                [1.2882952853460443, 0.7882952884758264, 0.1156062839099075])}
        outputNodes = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        outputDatapoints = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        TOL = 1.0E-12
        for group in outputGroups:
            groupName = group.getName()
            nodeset = outputNodes if groupName in ("cube", "square", "line") else outputDatapoints
            min, max = evaluate_field_nodeset_range(outputBodyCoordinates,
                                                    group.getFieldNodeGroup(nodeset).getNodesetGroup())
            assertAlmostEqualList(self, min, expectedBodyCoordinatesRange[groupName][0], TOL)
            assertAlmostEqualList(self, max, expectedBodyCoordinatesRange[groupName][1], TOL)

        # change some settings to test serialisation
        coordinatesField = dataEmbedder.getRootRegion().getFieldmodule().findFieldByName("fitted coordinates")
        self.assertTrue(coordinatesField.isValid())
        dataEmbedder.setMaterialCoordinatesField(coordinatesField)
        self.assertEqual("fitted coordinates", dataEmbedder.getMaterialCoordinatesField().getName())
        dataEmbedder.setGroupEmbed("line", False)
        dataEmbedder.setGroupEmbed("bottom", True)
        jsonString = dataEmbedder.encodeSettingsJSON()

        dataEmbedder2 = DataEmbedder(zincScaffoldFileName, zincFittedGeometryFileName, zincDataFileName)
        dataEmbedder2.decodeSettingsJSON(jsonString)
        dataEmbedder2.load()
        # check fields and group data automatically determined
        self.assertEqual("coordinates", dataEmbedder2.getDataCoordinatesField().getName())
        self.assertEqual("fitted coordinates", dataEmbedder2.getFittedCoordinatesField().getName())
        self.assertEqual("fitted coordinates", dataEmbedder2.getMaterialCoordinatesField().getName())
        self.assertEqual("marker", dataEmbedder2.getDataMarkerGroup().getName())
        groupNames = dataEmbedder2.getGroupNames()
        self.assertEqual(7, len(groupNames))
        self.assertTrue(dataEmbedder2.isGroupEmbed("cube"))
        self.assertTrue(dataEmbedder2.isGroupEmbed("square"))
        self.assertFalse(dataEmbedder2.isGroupEmbed("line"))
        self.assertTrue(dataEmbedder2.isGroupEmbed("ICN"))
        self.assertTrue(dataEmbedder2.isGroupEmbed("bottom"))
        self.assertFalse(dataEmbedder2.isGroupEmbed("marker"))
        self.assertFalse(dataEmbedder2.isGroupEmbed("tip 2"))


if __name__ == "__main__":
    unittest.main()
