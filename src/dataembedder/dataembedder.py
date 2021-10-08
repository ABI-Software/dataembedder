"""
Main class for fitting scaffolds.
"""

import json
import sys
from opencmiss.utils.zinc.field import get_group_list
from opencmiss.utils.zinc.general import ChangeManager
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field, FieldFindMeshLocation, FieldFiniteElement, FieldGroup
from opencmiss.zinc.region import Region
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE


class DataEmbedder:

    _embedToken = "embed"
    _dimensionToken = "dimension"
    _sizeToken = "size"

    def __init__(self, zincScaffoldFileName: str, zincFittedGeometryFileName, zincDataFileName: str):
        """
        :param zincScaffoldFileName: Name of zinc model file supplying full scaffold to embed in.
        :param zincFittedGeometryFileName: Name of zinc mode file defining fitted geometric state to embed data from.
        :param zincDataFileName: Name of zinc file supplying data in fitted state to embed in scaffold.
        """
        self._zincScaffoldFileName = zincScaffoldFileName
        self._zincFittedGeometryFileName = zincFittedGeometryFileName
        self._zincDataFileName = zincDataFileName
        self._context = Context("DataEmbedder")
        _, self._zincVersion = self._context.getVersion()
        self._logger = self._context.getLogger()
        self._rootRegion = None
        self._rootMesh = None
        self._rootBoundaryMesh = None
        self._dataRegion = None
        self._outputRegion = None
        self._dataCoordinatesField = None
        self._dataCoordinatesFieldName = None
        self._fittedCoordinatesField = None
        self._fittedCoordinatesFieldName = None
        self._materialCoordinatesField = None
        self._materialCoordinatesFieldName = None
        self._dataMarkerGroup = None
        self._dataMarkerGroupName = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        self._coordinatesArgumentField = None
        self._hostMaterialCoordinatesField = None
        self._diagnosticLevel = 0
        # groupData e.g. "groupName" -> { "embed": True, "dimension": 1, "size": 2 }
        # eventually , "TermID" : "UBERON:0000056"
        self._groupData = {}
        # client is now expected to call decodeSettingsJSON() if appropriate, then load()

    def decodeSettingsJSON(self, s: str):
        """
        Define DataEmbedder settings from JSON serialisation output by encodeSettingsJSON.
        :param s: String of JSON encoded embedder settings.
        """
        dct = json.loads(s)
        # field names are read (default to None), fields are found on load
        self._dataCoordinatesFieldName = dct.get("dataCoordinatesField")
        self._fittedCoordinatesFieldName = dct.get("fittedCoordinatesField")
        self._materialCoordinatesFieldName = dct.get("materialCoordinatesField")
        self._dataMarkerGroupName = dct.get("dataMarkerGroup")
        self._diagnosticLevel = dct["diagnosticLevel"]
        self._groupData = dct["groupData"]

    def encodeSettingsJSON(self) -> str:
        """
        :return: String JSON encoding of settings.
        """
        dct = {
            "dataCoordinatesField": self._dataCoordinatesFieldName,
            "fittedCoordinatesField": self._fittedCoordinatesFieldName,
            "materialCoordinatesField": self._materialCoordinatesFieldName,
            "dataMarkerGroup": self._dataMarkerGroupName,
            "diagnosticLevel": self._diagnosticLevel,
            "groupData": self._groupData
            }
        return json.dumps(dct, sort_keys=False, indent=4)

    def _clearFields(self):
        self._dataCoordinatesField = None
        self._fittedCoordinatesField = None
        self._materialCoordinatesField = None
        self._dataMarkerGroup = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        self._coordinatesArgumentField = None
        self._hostMaterialCoordinatesField = None

    @staticmethod
    def _findCoordinatesField(fieldmodule, fieldName: str, namePrefix: str = None) -> FieldFiniteElement:
        """
        Find Finite Element coordinates field, with the supplied name + optional prefix.
        :param fieldmodule: Fieldmodule to search in.
        :param fieldName: Suggested field name or None. Finds first coordinate field if not found or None.
        :param namePrefix: If supplied, added to existing names in the comparison if not already present,
        and if matched the prefix is added to the name. fieldName must start with the namePrefix. Prefix is
        not expected to contain a separator; if added to the name a space is added between it and the name.
        :return: Zinc FieldFiniteElement or None if not found.
        """
        assert (not fieldName) or (not namePrefix) or (0 == fieldName.find(namePrefix)), \
            "DataEmbedder._findCoordinatesField.  Field name does not contain prefix"
        coordinatesField = None
        fielditerator = fieldmodule.createFielditerator()
        field = fielditerator.next()
        while field.isValid():
            fieldFiniteElement = field.castFiniteElement()
            if fieldFiniteElement.isValid() and (field.getNumberOfComponents() <= 3) and field.isTypeCoordinate():
                if not fieldName:
                    coordinatesField = fieldFiniteElement
                    break
                thisFieldName = field.getName()
                if namePrefix and (0 != thisFieldName.find(namePrefix)):
                    thisFieldName = namePrefix + " " + thisFieldName
                if thisFieldName == fieldName:
                    coordinatesField = fieldFiniteElement
                    break
                if coordinatesField is None:
                    coordinatesField = fieldFiniteElement
            field = fielditerator.next()
        coordinatesFieldName = None
        if coordinatesField:
            coordinatesFieldName = coordinatesField.getName()
            if namePrefix and (0 != coordinatesFieldName.find(namePrefix)):
                coordinatesFieldName = namePrefix + " " + coordinatesFieldName
                coordinatesField.setName(coordinatesFieldName)
        if fieldName and ((coordinatesField is None) or (coordinatesFieldName != fieldName)):
            print("DataEmbedder. Did not find coordinates field of name " + fieldName, file=sys.stderr)
        return coordinatesField

    @staticmethod
    def _guessMaterialCoordinatesFieldName(fieldmodule):
        """
        Find likely material coordinate field based on largest group name + " coordinates" then
        ensure it exists.
        :param fieldmodule: Fieldmodule to search in.
        :return: Likely material coordinates field name (guaranteed to exist, but still needs to be checked
        for validity) or None if not found.
        """
        mesh = None
        for dimension in range(3, 0, -1):
            mesh = fieldmodule.findMeshByDimension(dimension)
            if mesh.getSize() > 0:
                break
        if mesh:
            largestGroupName = None
            largestSize = 0
            for group in get_group_list(fieldmodule):
                meshGroup = group.getFieldElementGroup(mesh).getMeshGroup()
                if meshGroup.isValid():
                    thisSize = meshGroup.getSize()
                    if thisSize > largestSize:
                        largestGroupName = group.getName()
                        largestSize = thisSize
            if largestGroupName:
                fieldName = largestGroupName + " coordinates"  # our material coordinate name convention
                if fieldmodule.findFieldByName(fieldName).isValid():
                    return fieldName
        return None

    def _discoverDataMarkerGroup(self):
        self._dataMarkerGroup = None
        self._dataMarkerNameField = None
        self._dataMarkerCoordinatesField = None
        dataMarkerGroup = self._dataRegion.getFieldmodule().findFieldByName(
            self._dataMarkerGroupName if self._dataMarkerGroupName else "marker").castGroup()
        self.setDataMarkerGroup(dataMarkerGroup if dataMarkerGroup.isValid() else None)

    def _buildEmbeddedMapFields(self):
        self._coordinatesArgumentField = None
        self._hostMaterialCoordinatesField = None
        if self._fittedCoordinatesField and self._materialCoordinatesField:
            rootFieldmodule = self._rootRegion.getFieldmodule()
            with ChangeManager(rootFieldmodule):
                self._coordinatesArgumentField = rootFieldmodule.createFieldArgumentReal(
                    self._fittedCoordinatesField.getNumberOfComponents())
                findMeshLocationField = rootFieldmodule.createFieldFindMeshLocation(
                    self._coordinatesArgumentField, self._fittedCoordinatesField, self._rootMesh)
                if self._rootMesh.getDimension() == 3:
                    # workaround for nearest around the boundary which doesn't work in 3D
                    # if no exact location found in root mesh, find nearest on the boundary mesh
                    findMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_EXACT)
                    hostExactMaterialCoordinatesField = rootFieldmodule.createFieldEmbedded(
                        self._materialCoordinatesField, findMeshLocationField)
                    hasFoundMeshLocation = rootFieldmodule.createFieldIsDefined(findMeshLocationField)
                    # want locations on the root mesh, but to search on the boundary mesh
                    findBoundaryMeshLocationField = rootFieldmodule.createFieldFindMeshLocation(
                        self._coordinatesArgumentField, self._fittedCoordinatesField, self._rootMesh)
                    findBoundaryMeshLocationField.setSearchMesh(self._rootBoundaryMesh)
                    findBoundaryMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                    hostBoundaryMaterialCoordinatesField = rootFieldmodule.createFieldEmbedded(
                        self._materialCoordinatesField, findBoundaryMeshLocationField)
                    valid1 = hostExactMaterialCoordinatesField.isValid()
                    valid2 = hostBoundaryMaterialCoordinatesField.isValid()
                    valid3 = hasFoundMeshLocation.isValid()
                    self._hostMaterialCoordinatesField = rootFieldmodule.createFieldIf(
                        hasFoundMeshLocation, hostExactMaterialCoordinatesField, hostBoundaryMaterialCoordinatesField)
                else:
                    findMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                    self._hostMaterialCoordinatesField = rootFieldmodule.createFieldEmbedded(
                        self._materialCoordinatesField, findMeshLocationField)
                assert self._hostMaterialCoordinatesField.isValid(), "Failed to create embedding map fields"

    def _buildGroupData(self):
        groupData = {}
        rootFieldmodule = self._rootRegion.getFieldmodule()
        dataFieldmodule = self._dataRegion.getFieldmodule()
        dataMesh = [None]
        maxDimension = 0
        for dimension in range(1, 4):
            dataMesh.append(dataFieldmodule.findMeshByDimension(dimension))
            if dataMesh[dimension].getSize() > 0:
                maxDimension = dimension
        # process regular groups
        # dimension 1-3 should use elements and nodes
        # dimension 0 should be datapoints
        datapoints = dataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        for group in get_group_list(dataFieldmodule):
            groupName = group.getName()
            # groups also in the root region are likely fitting contours or fiducial markers
            groupIsInRoot = rootFieldmodule.findFieldByName(groupName).castGroup().isValid()
            groupSize = 0
            groupDimension = 0
            for groupDimension in range(maxDimension, 0, -1):
                elementGroup = group.getFieldElementGroup(dataMesh[groupDimension])
                if elementGroup.isValid():
                    groupSize = elementGroup.getMeshGroup().getSize()
                    if groupSize > 0:
                        break
            if groupSize == 0:
                nodesetGroup = group.getFieldNodeGroup(datapoints).getNodesetGroup()
                if nodesetGroup.isValid():
                    groupSize = nodesetGroup.getSize()
            embed = not (groupIsInRoot or (groupSize == 0) or (group == self._dataMarkerGroup))
            groupData[groupName] = {
                self._embedToken: embed,
                self._dimensionToken: groupDimension,
                self._sizeToken: groupSize
            }
        # process data marker points making groups out of those with the same marker name
        if self._dataMarkerGroup and self._dataMarkerNameField:
            fieldcache = dataFieldmodule.createFieldcache()
            dataMarkerNodesetGroup = self._dataMarkerGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
            markerGroupData = {}
            lastMarkerName = None
            lastMarkerGroupDict = None
            nodeiter = dataMarkerNodesetGroup.createNodeiterator()
            node = nodeiter.next()
            while node.isValid():
                fieldcache.setNode(node)
                markerName = self._dataMarkerNameField.evaluateString(fieldcache)
                if markerName:
                    if markerName == lastMarkerName:
                        markerGroupDict = lastMarkerGroupDict
                    elif markerName in markerGroupData:
                        lastMarkerGroupDict = markerGroupDict = markerGroupData[markerName]
                        lastMarkerName = markerName
                    else:
                        groupIsInRoot = rootFieldmodule.findFieldByName(markerName).castGroup().isValid()
                        embed = not (groupIsInRoot or (markerName == self._dataMarkerGroupName))
                        lastMarkerGroupDict = markerGroupDict = {
                            self._embedToken: embed,
                            self._dimensionToken: 0,
                            self._sizeToken: 0
                        }
                        markerGroupData[markerName] = markerGroupDict
                        lastMarkerName = markerName
                    markerGroupDict[self._sizeToken] += 1
                node = nodeiter.next()
            markerGroupData.update(groupData)
            groupData = markerGroupData
        # transfer embed flag from existing groupData before replacing
        for groupName, groupDict in groupData.items():
            embed = self.isGroupEmbed(groupName) if (groupName in self._groupData) else None
            if embed is not None:
                groupDict[self._embedToken] = embed
        self._groupData = groupData

    def load(self):
        """
        Read model and data and define fields.
        Can call again to reset if inputs change.
        """
        self._clearFields()
        self._rootMesh = None
        self._rootBoundaryMesh = None
        self._rootRegion = self._context.createRegion()
        self._dataRegion = self._rootRegion.createChild("data")
        rootFieldmodule = self._rootRegion.getFieldmodule()
        with ChangeManager(rootFieldmodule):
            result = self._rootRegion.readFile(self._zincFittedGeometryFileName)
            assert result == RESULT_OK, "Failed to load fitted geometry file" + str(self._zincFittedGeometryFileName)
            self._fittedCoordinatesField =\
                self._findCoordinatesField(rootFieldmodule, self._fittedCoordinatesFieldName, namePrefix="fitted")
            if self._fittedCoordinatesField:
                self._fittedCoordinatesFieldName = self._fittedCoordinatesField.getName()

            result = self._rootRegion.readFile(self._zincScaffoldFileName)
            assert result == RESULT_OK, "Failed to load scaffold file" + str(self._zincScaffoldFileName)
            if not self._materialCoordinatesFieldName:
                self._materialCoordinatesFieldName = self._guessMaterialCoordinatesFieldName(rootFieldmodule)
            self._materialCoordinatesField =\
                self._findCoordinatesField(rootFieldmodule, self._materialCoordinatesFieldName)
            if self._materialCoordinatesField:
                self._materialCoordinatesFieldName = self._materialCoordinatesField.getName()

            # get highest dimension mesh in root, and its boundary mesh (dimension - 1)
            for dimension in range(3, 0, -1):
                rootMesh = rootFieldmodule.findMeshByDimension(dimension)
                if rootMesh.getSize() > 0:
                    self._rootMesh = rootMesh
                    if dimension > 1:
                        self._rootBoundaryMesh = rootFieldmodule.findMeshByDimension(dimension - 1)
                    break
            self._buildEmbeddedMapFields()

        dataFieldmodule = self._dataRegion.getFieldmodule()
        result = self._dataRegion.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file" + str(self._zincDataFileName)
        self._dataCoordinatesField = self._findCoordinatesField(dataFieldmodule, self._dataCoordinatesFieldName)
        if self._dataCoordinatesField:
            self._dataCoordinatesFieldName = self._dataCoordinatesField.getName()

        self._discoverDataMarkerGroup()
        self._buildGroupData()

    def getContext(self) -> Context:
        return self._context

    def getRootRegion(self) -> Region:
        """
        Get root region where the host scaffold is loaded.
        :return: Zinc Region
        """
        return self._rootRegion

    def getDataRegion(self) -> Region:
        """
        Get the child data region where the embedded data is loaded.
        :return: Zinc Region
        """
        return self._dataRegion

    def getOutputRegion(self) -> Region:
        """
        Get the child region where the output is created. Must call generateOutput first.
        :return: Zinc Region
        """
        return self._dataRegion

    def getDataCoordinatesField(self) -> FieldFiniteElement:
        """
        Get the field on the data region giving the coordinates to find embedded locations from.
        """
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField: FieldFiniteElement):
        """
        Set the field on the data region giving the coordinates to find embedded locations from.
        :param dataCoordinatesField: Data coordinates field defined on the data region.
        """
        if dataCoordinatesField == self._dataCoordinatesField:
            return
        finiteElementField = dataCoordinatesField.castFiniteElement() if dataCoordinatesField else None
        assert ((dataCoordinatesField is not None) and
            (dataCoordinatesField.getFieldmodule().getRegion() == self._dataRegion) and
            finiteElementField.isValid() and (dataCoordinatesField.getNumberOfComponents() <= 3))
        self._dataCoordinatesField = finiteElementField
        self._dataCoordinatesFieldName = dataCoordinatesField.getName()

    def getFittedCoordinatesField(self):
        """
        Get the field on the root/scaffold region giving the fitted coordinates the data coordinates are
        relative to.
        """
        return self._fittedCoordinatesField

    def setFittedCoordinatesField(self, fittedCoordinatesField: FieldFiniteElement):
        """
        Set the field on the root/scaffold region giving the fitted coordinates the data coordinates are
        relative to.
        :param fittedCoordinatesField: Fitted coordinates field defined on the root region.
        """
        if fittedCoordinatesField == self._fittedCoordinatesField:
            return
        finiteElementField = fittedCoordinatesField.castFiniteElement() if fittedCoordinatesField else None
        assert ((fittedCoordinatesField is not None) and
            (fittedCoordinatesField.getFieldmodule().getRegion() == self._rootRegion) and
            finiteElementField.isValid() and (fittedCoordinatesField.getNumberOfComponents() <= 3))
        self._fittedCoordinatesField = finiteElementField
        self._fittedCoordinatesFieldName = fittedCoordinatesField.getName()

    def getMaterialCoordinatesField(self):
        """
        Get the field on the root/scaffold region giving the material coordinates embedded locations need to supply.
        """
        return self._materialCoordinatesField

    def setMaterialCoordinatesField(self, materialCoordinatesField: FieldFiniteElement):
        """
        Set the field on the root/scaffold region giving the material coordinates embedded locations need to supply.
        :param materialCoordinatesField: Material coordinates field defined on the root region.
        """
        if materialCoordinatesField == self._materialCoordinatesField:
            return
        finiteElementField = materialCoordinatesField.castFiniteElement() if materialCoordinatesField else None
        assert ((materialCoordinatesField is not None) and
            (materialCoordinatesField.getFieldmodule().getRegion() == self._rootRegion) and
            finiteElementField.isValid() and (materialCoordinatesField.getNumberOfComponents() <= 3))
        self._materialCoordinatesField = finiteElementField
        self._materialCoordinatesFieldName = materialCoordinatesField.getName()

    def getDataMarkerGroup(self):
        return self._dataMarkerGroup

    def setDataMarkerGroup(self, dataMarkerGroup: FieldGroup):
        """
        Set the marker group from which point data is extracted via its name field. The name field and coordinates
        field are automatically discovered from the group by looking at the datapoints in it.
        :param dataMarkerGroup: Marker group in data subregion, or None. Both fiducial markers and embedded point
        data will be in this group. A stored string field on the points gives the group name.
        """
        assert (dataMarkerGroup is None) or (dataMarkerGroup.castGroup().isValid() and
               (dataMarkerGroup.getFieldmodule().getRegion() == self._dataRegion))
        self._dataMarkerGroup = None
        self._dataMarkerGroupName = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        if not dataMarkerGroup:
            return
        self._dataMarkerGroup = dataMarkerGroup.castGroup()
        self._dataMarkerGroupName = self._dataMarkerGroup.getName()
        dataFieldmodule = self._dataRegion.getFieldmodule()
        datapoints = dataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        dataMarkerNodesetGroup = self._dataMarkerGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
        if dataMarkerNodesetGroup.isValid():
            node = dataMarkerNodesetGroup.createNodeiterator().next()
            if node.isValid():
                fieldcache = dataFieldmodule.createFieldcache()
                fieldcache.setNode(node)
                # coordinates is likely the same as for other data fields
                if self._dataCoordinatesField and self._dataCoordinatesField.isDefinedAtLocation(fieldcache):
                    self._dataMarkerCoordinatesField = self._dataCoordinatesField
                fielditer = dataFieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._dataMarkerCoordinatesField) and field.castFiniteElement().isValid():
                            self._dataMarkerCoordinatesField = field
                        elif (not self._dataMarkerNameField) and (field.castStoredString().isValid()):
                            self._dataMarkerNameField = field
                    field = fielditer.next()
        if (self._diagnosticLevel > 0) and (not self._dataMarkerCoordinatesField) or (not self._dataMarkerNameField):
            print("Data marker group", self._dataMarkerGroupName, "is empty or has no coordinates or name field")

    def getDiagnosticLevel(self) -> int:
        return self._diagnosticLevel

    def setDiagnosticLevel(self, diagnosticLevel: int):
        """
        :param diagnosticLevel: 0 = no diagnostic messages. 1+ = Information and warning messages.
        """
        assert diagnosticLevel >= 0
        self._diagnosticLevel = diagnosticLevel

    def getGroupNames(self):
        return self._groupData.keys()

    def hasGroup(self, groupName: str) -> bool:
        """
        Query whether group of name exists in data.
        :param groupName: Name of the group
        :return: True if group exists, otherwise False.
        """
        return groupName in self._groupData

    def getGroupDimension(self, groupName: str) -> bool:
        """
        Get highest dimension in objects of data group of name.
        :param groupName: Name of the group
        :return: Dimension >= 0, or -1 if group not found.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._dimensionToken]
        print("DataEmbedder getGroupDimension: no group of name " + str(groupName), file=sys.stderr)
        return 0

    def isGroupEmbed(self, groupName: str) -> bool:
        """
        Query whether data will be embedded for the group.
        :param groupName: Name of the group
        :return: True if group is to be embedded, otherwise False.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._embedToken]
        print("DataEmbedder isGroupEmbed: no group of name " + str(groupName), file=sys.stderr)
        return False

    def setGroupEmbed(self, groupName: str, embed: bool):
        """
        Set whether to embed data for the group.
        :param groupName: Name of the group
        :param embed: True to embed group data, False to not embed.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            groupDict[self._embedToken] = embed
            return
        print("DataEmbedder groupSetEmbed: no group of name " + str(groupName), file=sys.stderr)

    def getGroupSize(self, groupName: str) -> bool:
        """
        Get number of objects in data group of name of its highest dimension.
        :param groupName: Name of the group
        :return: Size > 0, or -1 if group not found.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._sizeToken]
        print("DataEmbedder getGroupSize: no group of name " + str(groupName), file=sys.stderr)
        return -1

    def generateOutput(self) -> Region:
        """
        Generate embedded data from the groups with their embed flag set.
        :return: Zinc Region containing embedded output data. Regenerated every time this is called.
        """
        if self._outputRegion:
            self._rootRegion.removeChild(self._outputRegion)
        self._outputRegion = self._rootRegion.createChild("output")
        result = self._outputRegion.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file into output" + str(self._zincDataFileName)
        outputFieldmodule = self._outputRegion.getFieldmodule()
        outputMesh = [None]
        for dimension in range(1, 4):
            outputMesh.append(outputFieldmodule.findMeshByDimension(dimension))
        outputNodes = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        outputDatapoints = outputFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        with ChangeManager(outputFieldmodule):
            # make a group containing all the objects we want to keep
            embedGroup = outputFieldmodule.createFieldGroup()
            embedMeshGroup = [None]
            for dimension in range(1, 4):
                embedMeshGroup.append(embedGroup.createFieldElementGroup(outputMesh[dimension]).getMeshGroup())
            embedNodeGroup = embedGroup.createFieldNodeGroup(outputNodes).getNodesetGroup()
            embedDataGroup = embedGroup.createFieldNodeGroup(outputDatapoints).getNodesetGroup()
            outputDataMarkerGroup = None
            outputDataMarkerNameField = None
            if self._dataMarkerGroup:
                outputDataMarkerGroup = outputFieldmodule.findFieldByName(self._dataMarkerGroupName).castGroup()
                outputDataMarkerNameField =\
                    outputFieldmodule.findFieldByName(self._dataMarkerNameField.getName())
            for groupName, groupDict in self._groupData.items():
                group = outputFieldmodule.findFieldByName(groupName).castGroup()
                if groupDict[self._embedToken]:
                    groupDimension = groupDict[self._dimensionToken]
                    if group.isValid():
                        for dimension in range(groupDimension, 0, -1):
                            embedMeshGroup[dimension].addElementsConditional(group)
                        embedNodeGroup.addNodesConditional(group)
                    else:
                        assert outputDataMarkerGroup and outputDataMarkerNameField,\
                            "Missing data marker group and fields"
                        # marker points: make zinc group of datapoints with the same name
                        group = outputFieldmodule.createFieldGroup()
                        if group.setName(groupName) != RESULT_OK:
                            print("Failed to set marker group name " + groupName + "; skipping", file=sys.stderr)
                            del group
                            continue
                        group.setManaged(True)
                        dataGroup = group.createFieldNodeGroup(outputDatapoints).getNodesetGroup()
                        fieldcache = outputFieldmodule.createFieldcache()
                        outputDataMarkerDataGroup =\
                            outputDataMarkerGroup.getFieldNodeGroup(outputDatapoints).getNodesetGroup()
                        nodeiter = outputDataMarkerDataGroup.createNodeiterator()
                        node = nodeiter.next()
                        while node.isValid():
                            fieldcache.setNode(node)
                            markerName = outputDataMarkerNameField.evaluateString(fieldcache)
                            if markerName == groupName:
                                dataGroup.addNode(node)
                            node = nodeiter.next()
                    # add data points in group for both cases
                    embedDataGroup.addNodesConditional(group)
                elif groupName != self._dataMarkerGroupName:
                    # remove non-embedding groups from output, except marker group
                    if group.isValid():
                        group.setManaged(False)
                        del group  # so not keeping a handle to a group being removed
            # destroy everything not in embedGroup and remove embedGroup and any other groups not being embedded
            notEmbedGroup = outputFieldmodule.createFieldNot(embedGroup)
            for dimension in range(1, 4):
                outputMesh[dimension].destroyElementsConditional(notEmbedGroup)
            outputNodes.destroyNodesConditional(notEmbedGroup)
            outputDatapoints.destroyNodesConditional(notEmbedGroup)
            del notEmbedGroup
            del embedGroup
            del embedMeshGroup
            del embedNodeGroup
            del embedDataGroup
            # define material coordinates same as data coordinates by writing to memory buffer, renaming and re-reading
            # handle case where marker coordinates is a different field
            differentMarkerCoordinates = self._dataMarkerCoordinatesField and \
                (not (self._dataMarkerCoordinatesField == self._dataCoordinatesField))
            outputDataCoordinatesFieldNames = [self._dataCoordinatesFieldName]
            if differentMarkerCoordinates:
                outputDataCoordinatesFieldNames.append(self._dataMarkerCoordinatesField.getName())
            outputDataCoordinatesFields =\
                [outputFieldmodule.findFieldByName(fieldName).castFiniteElement()
                 for fieldName in outputDataCoordinatesFieldNames]
            sir = self._outputRegion.createStreaminformationRegion()
            srm = sir.createStreamresourceMemory()
            sir.setResourceFieldNames(srm, outputDataCoordinatesFieldNames)
            self._outputRegion.write(sir)
            result, buffer = srm.getBuffer()
            assert result == RESULT_OK, "Failed to write data coordinates to memory"
            # rename fields in the buffer
            for fieldName in outputDataCoordinatesFieldNames:
                buffer = buffer.replace(bytes(fieldName + ", coordinate", "utf-8"),
                                        bytes(self._materialCoordinatesFieldName + ", coordinate", "utf-8"))
            sir = self._outputRegion.createStreaminformationRegion()
            sir.createStreamresourceMemoryBuffer(buffer)
            result = self._outputRegion.read(sir)
            assert result == RESULT_OK, "Failed to define output material coordinates from memory buffer"
            outputMaterialCoordinatesField =\
                outputFieldmodule.findFieldByName(self._materialCoordinatesFieldName).castFiniteElement()
            assert outputMaterialCoordinatesField.isValid(), "Failed to define output materiel coordinates field"

            # now do the embedding: evaluate material coordinates in output region
            for field in outputDataCoordinatesFields:
                applyField = outputFieldmodule.createFieldApply(self._hostMaterialCoordinatesField)
                result = applyField.setBindArgumentSourceField(self._coordinatesArgumentField, field)
                assert result == RESULT_OK, "Failed to set bind argument source field in output"
                fieldassignment = outputMaterialCoordinatesField.createFieldassignment(applyField)
                for nodeset in (outputNodes, outputDatapoints):
                    fieldassignment.setNodeset(nodeset)
                    result = fieldassignment.assign()
                    assert result in (RESULT_OK, RESULT_WARNING_PART_DONE), "Failed to assign material coordinates"

        return self._outputRegion
