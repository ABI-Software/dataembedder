import os
import unittest
from opencmiss.utils.zinc.field import get_group_list
from opencmiss.utils.zinc.finiteelement import evaluate_field_nodeset_range
from opencmiss.zinc.field import Field
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
        groupNames = dataEmbedder.getDataGroupNames()
        self.assertEqual(8, len(groupNames))
        self.assertTrue(dataEmbedder.isDataGroupEmbed("cube"))
        self.assertTrue(dataEmbedder.isDataGroupEmbed("square"))
        self.assertTrue(dataEmbedder.isDataGroupEmbed("line"))
        self.assertTrue(dataEmbedder.isDataGroupEmbed("ICN"))
        self.assertTrue(dataEmbedder.isDataGroupEmbed("nerve"))
        self.assertFalse(dataEmbedder.isDataGroupEmbed("bottom"))
        self.assertFalse(dataEmbedder.isDataGroupEmbed("marker"))
        self.assertFalse(dataEmbedder.isDataGroupEmbed("tip 2"))
        self.assertFalse(dataEmbedder.hasDataGroup("tip 1"))
        self.assertEqual(3, dataEmbedder.getDataGroupDimension("cube"))
        self.assertEqual(2, dataEmbedder.getDataGroupDimension("square"))
        self.assertEqual(1, dataEmbedder.getDataGroupDimension("line"))
        self.assertEqual(0, dataEmbedder.getDataGroupDimension("ICN"))
        self.assertEqual(1, dataEmbedder.getDataGroupDimension("nerve"))
        self.assertEqual(1, dataEmbedder.getDataGroupSize("cube"))
        self.assertEqual(1, dataEmbedder.getDataGroupSize("square"))
        self.assertEqual(1, dataEmbedder.getDataGroupSize("line"))
        self.assertEqual(4, dataEmbedder.getDataGroupSize("ICN"))
        self.assertEqual(3, dataEmbedder.getDataGroupSize("nerve"))
        # test setting and unsetting embed flag
        dataEmbedder.setDataGroupEmbed("bottom", True)
        self.assertTrue(dataEmbedder.isDataGroupEmbed("bottom"))
        dataEmbedder.setDataGroupEmbed("bottom", False)
        self.assertFalse(dataEmbedder.isDataGroupEmbed("bottom"))

        outputRegion = dataEmbedder.generateOutput()
        self.assertTrue(outputRegion.isValid())

        # check output
        # note that the nerve group coordinates were partially outside the host domain, but its
        # body coordinates are forced to the nearest locations on the boundary where outside
        outputFieldmodule = outputRegion.getFieldmodule()
        outputBodyCoordinates = outputFieldmodule.findFieldByName("body coordinates").castFiniteElement()
        self.assertTrue(outputBodyCoordinates.isValid())
        outputGroups = get_group_list(outputFieldmodule)
        self.assertEqual(6, len(outputGroups))
        expectedBodyCoordinatesRange = {
            'ICN': ([0.4234094227546622, 0.48078031509540764, 0.24053424057686418],
                [0.4714586372862805, 0.5096098437995359, 0.2789736122606712]),
            'cube': ([0.13511413479016324, 0.21170471114299036, 0.11560628129407298],
                [0.7117047108476393, 0.7882952895792783, 0.6921968588706179]),
            'line': ([1.2882952842495057, 0.21170471154682882, 0.11560628082571832],
                [1.8648858589797928, 0.21170471264989535, 0.11560628106112401]),
            'marker': ([0.4234094227546622, 0.48078031509540764, 0.24053424057686418],
                [0.4714586372862805, 0.5096098437995359, 0.2789736122606712]),
            'nerve': ([0.6828751816364316, 0.11560627908676875, 0.8843937155748031],
                [1.4324429293766905, 0.7882952832850739, 1.0]),
            'square': ([0.7117047095185545, 0.21170471264989535, 0.11560628106112401],
                [1.2882952853460443, 0.7882952884758264, 0.1156062839099075])}
        outputNodes = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        outputDatapoints = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        TOL = 1.0E-12
        for group in outputGroups:
            groupName = group.getName()
            nodeset = outputNodes if groupName in ("cube", "square", "line", "nerve") else outputDatapoints
            minX, maxX = evaluate_field_nodeset_range(outputBodyCoordinates,
                                                    group.getFieldNodeGroup(nodeset).getNodesetGroup())
            assertAlmostEqualList(self, minX, expectedBodyCoordinatesRange[groupName][0], TOL)
            assertAlmostEqualList(self, maxX, expectedBodyCoordinatesRange[groupName][1], TOL)

        # change some settings to test serialisation
        coordinatesField = dataEmbedder.getHostRegion().getFieldmodule().findFieldByName("fitted coordinates")
        self.assertTrue(coordinatesField.isValid())
        dataEmbedder.setMaterialCoordinatesField(coordinatesField)
        self.assertEqual("fitted coordinates", dataEmbedder.getMaterialCoordinatesField().getName())
        dataEmbedder.setDataGroupEmbed("line", False)
        dataEmbedder.setDataGroupEmbed("bottom", True)
        jsonString = dataEmbedder.encodeSettingsJSON()

        dataEmbedder2 = DataEmbedder(zincScaffoldFileName, zincFittedGeometryFileName, zincDataFileName)
        dataEmbedder2.decodeSettingsJSON(jsonString)
        dataEmbedder2.load()
        # check fields and group data automatically determined
        self.assertEqual("coordinates", dataEmbedder2.getDataCoordinatesField().getName())
        self.assertEqual("fitted coordinates", dataEmbedder2.getFittedCoordinatesField().getName())
        self.assertEqual("fitted coordinates", dataEmbedder2.getMaterialCoordinatesField().getName())
        self.assertEqual("marker", dataEmbedder2.getDataMarkerGroup().getName())
        groupNames = dataEmbedder2.getDataGroupNames()
        self.assertEqual(8, len(groupNames))
        self.assertTrue(dataEmbedder2.isDataGroupEmbed("cube"))
        self.assertTrue(dataEmbedder2.isDataGroupEmbed("square"))
        self.assertFalse(dataEmbedder2.isDataGroupEmbed("line"))
        self.assertTrue(dataEmbedder2.isDataGroupEmbed("ICN"))
        self.assertTrue(dataEmbedder2.isDataGroupEmbed("nerve"))
        self.assertTrue(dataEmbedder2.isDataGroupEmbed("bottom"))
        self.assertFalse(dataEmbedder2.isDataGroupEmbed("marker"))
        self.assertFalse(dataEmbedder2.isDataGroupEmbed("tip 2"))


if __name__ == "__main__":
    unittest.main()
