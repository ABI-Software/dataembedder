"""
Main class for fitting scaffolds.
"""

import json
import sys
from opencmiss.utils.zinc.field import get_group_list, get_unique_field_name
from opencmiss.utils.zinc.general import ChangeManager, HierarchicalChangeManager
from opencmiss.zinc.context import Context
from opencmiss.zinc.element import Mesh, MeshGroup
from opencmiss.zinc.field import Field, FieldFindMeshLocation, FieldFiniteElement, FieldGroup
from opencmiss.zinc.region import Region
from opencmiss.zinc.result import RESULT_ERROR_NOT_FOUND, RESULT_OK, RESULT_WARNING_PART_DONE


class DataEmbedder:

    _embedToken = "embed"
    _dimensionToken = "dimension"
    _sizeToken = "size"

    def __init__(self, zincScaffoldFileName: str, zincFittedGeometryFileName, zincDataFileName: str):
        """
        :param zincScaffoldFileName: Name of zinc model file supplying full scaffold to embed in.
        :param zincFittedGeometryFileName: Name of zinc mode file defining fitted geometry of the scaffold to
        calculate embedding in.
        :param zincDataFileName: Name of zinc file supplying data in fitted state to embed in scaffold.
        """
        self._zincScaffoldFileName = zincScaffoldFileName
        self._zincFittedGeometryFileName = zincFittedGeometryFileName
        self._zincDataFileName = zincDataFileName
        self._context = Context("DataEmbedder")
        _, self._zincVersion = self._context.getVersion()
        self._logger = self._context.getLogger()
        self._hostRegion = None
        self._dataRegion = None
        self._outputDataRegion = None
        self._hostMesh = None
        self._hostBoundaryMesh = None
        self._fittedGroup = None
        self._fittedMeshGroup = None
        self._fittedBoundaryGroup = None
        self._fittedBoundaryMeshGroup = None
        self._fittedCoordinatesField = None
        self._fittedCoordinatesFieldName = None
        self._materialCoordinatesField = None
        self._materialCoordinatesFieldName = None
        self._hostMarkerGroup = None
        self._hostMarkerGroupName = None
        self._hostMarkerLocationField = None
        self._hostMarkerNameField = None
        self._hostProjectionGroup = None
        self._hostProjectionGroupName = None
        self._hostProjectionMeshGroup = None
        self._dataCoordinatesField = None
        self._dataCoordinatesFieldName = None
        self._dataMarkerGroup = None
        self._dataMarkerGroupName = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        self._coordinatesArgumentField = None
        self._hostFindMaterialCoordinatesField = None
        self._outputDataMaterialCoordinatesField = None
        self._diagnosticLevel = 0
        self._needGenerateOutput = True
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
        self._fittedCoordinatesFieldName = dct.get("fittedCoordinatesField")
        self._materialCoordinatesFieldName = dct.get("materialCoordinatesField")
        self._hostMarkerGroupName = dct.get("hostMarkerGroup")
        self._hostProjectionGroupName = dct.get("hostProjectionGroup")
        self._dataCoordinatesFieldName = dct.get("dataCoordinatesField")
        self._dataMarkerGroupName = dct.get("dataMarkerGroup")
        self._diagnosticLevel = dct["diagnosticLevel"]
        self._groupData = dct["groupData"]

    def encodeSettingsJSON(self) -> str:
        """
        :return: String JSON encoding of settings.
        """
        dct = {
            "fittedCoordinatesField": self._fittedCoordinatesFieldName,
            "materialCoordinatesField": self._materialCoordinatesFieldName,
            "hostMarkerGroup": self._hostMarkerGroupName,
            "hostProjectionGroup": self._hostProjectionGroupName,
            "dataCoordinatesField": self._dataCoordinatesFieldName,
            "dataMarkerGroup": self._dataMarkerGroupName,
            "diagnosticLevel": self._diagnosticLevel,
            "groupData": self._groupData
            }
        return json.dumps(dct, sort_keys=False, indent=4)

    def _clearFields(self):
        self._fittedGroup = None
        self._fittedMeshGroup = None
        self._fittedBoundaryGroup = None
        self._fittedBoundaryMeshGroup = None
        self._fittedCoordinatesField = None
        self._materialCoordinatesField = None
        self._hostMarkerGroup = None
        self._hostMarkerLocationField = None
        self._hostMarkerNameField = None
        self._hostProjectionGroup = None
        self._hostProjectionMeshGroup = None
        self._dataCoordinatesField = None
        self._dataMarkerGroup = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        self._coordinatesArgumentField = None
        self._hostFindMaterialCoordinatesField = None
        self._outputDataMaterialCoordinatesField = None

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

    def _discoverHostMarkerGroup(self):
        self._hostMarkerGroup = None
        self._hostMarkerLocationField = None
        self._hostMarkerNameField = None
        hostMarkerGroup = self._hostRegion.getFieldmodule().findFieldByName(
            self._hostMarkerGroupName if self._hostMarkerGroupName else "marker").castGroup()
        if not hostMarkerGroup.isValid():
            hostMarkerGroup = None
        self.setHostMarkerGroup(hostMarkerGroup)

    def _discoverHostProjectionGroup(self):
        self._hostProjectionGroup = None
        self._hostProjectionMeshGroup = None
        hostProjectionGroup = None
        if self._hostProjectionGroupName:
            hostProjectionGroup = self._hostRegion.getFieldmodule().findFieldByName(self._hostProjectionGroupName)
            if not hostProjectionGroup.isValid():
                hostProjectionGroup = None
        self.setHostProjectionGroup(hostProjectionGroup)

    def _discoverDataMarkerGroup(self):
        self._dataMarkerGroup = None
        self._dataMarkerNameField = None
        self._dataMarkerCoordinatesField = None
        dataMarkerGroup = self._dataRegion.getFieldmodule().findFieldByName(
            self._dataMarkerGroupName if self._dataMarkerGroupName else "marker").castGroup()
        self.setDataMarkerGroup(dataMarkerGroup if dataMarkerGroup.isValid() else None)

    def _buildEmbeddedMapFields(self):
        self._coordinatesArgumentField = None
        self._hostFindMaterialCoordinatesField = None
        if self._fittedCoordinatesField and self._materialCoordinatesField:
            hostFieldmodule = self._hostRegion.getFieldmodule()
            with ChangeManager(hostFieldmodule):
                self._coordinatesArgumentField = hostFieldmodule.createFieldArgumentReal(
                    self._fittedCoordinatesField.getNumberOfComponents())
                findMeshLocationField = hostFieldmodule.createFieldFindMeshLocation(
                    self._coordinatesArgumentField, self._fittedCoordinatesField, self._hostMesh)
                searchMesh = self._hostProjectionMeshGroup if self._hostProjectionMeshGroup else self._fittedMeshGroup
                findMeshLocationField.setSearchMesh(searchMesh)
                findMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                self._hostFindMaterialCoordinatesField = hostFieldmodule.createFieldEmbedded(
                    self._materialCoordinatesField, findMeshLocationField)
                assert self._hostFindMaterialCoordinatesField.isValid(), "Failed to create embedding map fields"

    def _getHostMarkerNames(self):
        """
        Get marker names in scaffold since these may not have an associated zinc group.
        :return: Set of distinct marker names in the host scaffold.
        """
        hostMarkerNames = set()
        if self._hostMarkerGroup and self._hostMarkerNameField:
            hostFieldmodule = self._hostRegion.getFieldmodule()
            fieldcache = hostFieldmodule.createFieldcache()
            hostNodes = hostFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            hostMarkerNodesetGroup = self._hostMarkerGroup.getFieldNodeGroup(hostNodes).getNodesetGroup()
            nodeiter = hostMarkerNodesetGroup.createNodeiterator()
            node = nodeiter.next()
            while node.isValid():
                fieldcache.setNode(node)
                markerName = self._hostMarkerNameField.evaluateString(fieldcache)
                if markerName:
                    hostMarkerNames.add(markerName)
                node = nodeiter.next()
        return hostMarkerNames

    def _buildDataGroups(self):
        groupData = {}
        hostFieldmodule = self._hostRegion.getFieldmodule()
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
            # groups also in the host region are likely fitting contours or fiducial markers
            groupIsInHost = hostFieldmodule.findFieldByName(groupName).castGroup().isValid()
            groupSize = 0
            groupDimension = 0
            for dimension in range(maxDimension, 0, -1):
                elementGroup = group.getFieldElementGroup(dataMesh[dimension])
                if elementGroup.isValid():
                    groupSize = elementGroup.getMeshGroup().getSize()
                    if groupSize > 0:
                        groupDimension = dimension
                        break
            if groupSize == 0:
                nodesetGroup = group.getFieldNodeGroup(datapoints).getNodesetGroup()
                if nodesetGroup.isValid():
                    groupSize = nodesetGroup.getSize()
            embed = not (groupIsInHost or (groupSize == 0) or (group == self._dataMarkerGroup))
            groupData[groupName] = {
                self._embedToken: embed,
                self._dimensionToken: groupDimension,
                self._sizeToken: groupSize
            }

        # process data marker points making groups out of those with the marker names not present in the host
        if self._dataMarkerGroup and self._dataMarkerNameField:
            fieldcache = dataFieldmodule.createFieldcache()
            hostMarkerNames = self._getHostMarkerNames()
            dataMarkerNodesetGroup = self._dataMarkerGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
            markerGroupData = {}
            nodeiter = dataMarkerNodesetGroup.createNodeiterator()
            node = nodeiter.next()
            while node.isValid():
                fieldcache.setNode(node)
                markerName = self._dataMarkerNameField.evaluateString(fieldcache)
                if markerName:
                    markerGroupDict = markerGroupData.get(markerName)
                    if markerGroupDict:
                        markerGroupDict[self._sizeToken] += 1
                    else:
                        groupIsInHost = hostFieldmodule.findFieldByName(markerName).castGroup().isValid() or \
                                        (markerName in hostMarkerNames)
                        embed = not groupIsInHost
                        markerGroupData[markerName] = markerGroupDict = {
                            self._embedToken: embed,
                            self._dimensionToken: 0,
                            self._sizeToken: 1
                        }
                    node = nodeiter.next()
            for markerName, markerGroupDict in markerGroupData.items():
                if (markerName not in groupData) and markerGroupDict[self._embedToken]:
                    groupData[markerName] = markerGroupDict
        # transfer embed flag from existing groupData before replacing
        for groupName, groupDict in groupData.items():
            embed = self.isDataGroupEmbed(groupName) if (groupName in self._groupData) else None
            if embed is not None:
                groupDict[self._embedToken] = embed
        self._groupData = groupData

    def load(self):
        """
        Read model and data and define fields.
        Can call again to reset if inputs change.
        """
        self._clearFields()
        self._hostMesh = None
        self._hostBoundaryMesh = None
        self._hostRegion = self._context.createRegion()
        with HierarchicalChangeManager(self._hostRegion):
            self._dataRegion = self._hostRegion.createChild("data")
            hostFieldmodule = self._hostRegion.getFieldmodule()
            result = self._hostRegion.readFile(self._zincFittedGeometryFileName)
            assert result == RESULT_OK, "Failed to load fitted geometry file" + str(self._zincFittedGeometryFileName)
            self._fittedCoordinatesField =\
                self._findCoordinatesField(hostFieldmodule, self._fittedCoordinatesFieldName, namePrefix="fitted")
            if self._fittedCoordinatesField:
                self._fittedCoordinatesFieldName = self._fittedCoordinatesField.getName()

            # get highest dimension mesh in host, and its boundary mesh (dimension - 1)
            for dimension in range(3, 0, -1):
                hostMesh = hostFieldmodule.findMeshByDimension(dimension)
                if hostMesh.getSize() > 0:
                    self._hostMesh = hostMesh
                    if dimension > 1:
                        self._hostBoundaryMesh = hostFieldmodule.findMeshByDimension(dimension - 1)
                    break

            # make group from elements where fitted field is defined, and its boundary
            self._fittedGroup = hostFieldmodule.createFieldGroup()
            self._fittedGroup.setName("fitted")
            self._fittedGroup.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
            self._fittedMeshGroup = self._fittedGroup.createFieldElementGroup(self._hostMesh).getMeshGroup()
            self._fittedMeshGroup.addElementsConditional(
                hostFieldmodule.createFieldIsDefined(self._fittedCoordinatesField) if self._fittedCoordinatesField else
                hostFieldmodule.createFieldConstant(1.0))
            if self._hostBoundaryMesh:
                self._fittedBoundaryGroup = hostFieldmodule.createFieldGroup()
                self._fittedBoundaryGroup.setName("fitted boundary")
                self._fittedBoundaryGroup.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
                self._fittedBoundaryMeshGroup =\
                    self._fittedBoundaryGroup.createFieldElementGroup(self._hostBoundaryMesh).getMeshGroup()
                self._fittedBoundaryMeshGroup.addElementsConditional(
                    hostFieldmodule.createFieldAnd(self._fittedGroup, hostFieldmodule.createFieldIsExterior()))

            result = self._hostRegion.readFile(self._zincScaffoldFileName)
            assert result == RESULT_OK, "Failed to load scaffold file" + str(self._zincScaffoldFileName)
            if not self._materialCoordinatesFieldName:
                self._materialCoordinatesFieldName = self._guessMaterialCoordinatesFieldName(hostFieldmodule)
            self._materialCoordinatesField =\
                self._findCoordinatesField(hostFieldmodule, self._materialCoordinatesFieldName)
            if self._materialCoordinatesField:
                self._materialCoordinatesFieldName = self._materialCoordinatesField.getName()

            dataFieldmodule = self._dataRegion.getFieldmodule()
            result = self._dataRegion.readFile(self._zincDataFileName)
            assert result == RESULT_OK, "Failed to load data file" + str(self._zincDataFileName)
            self._dataCoordinatesField = self._findCoordinatesField(dataFieldmodule, self._dataCoordinatesFieldName)
            if self._dataCoordinatesField:
                self._dataCoordinatesFieldName = self._dataCoordinatesField.getName()

            self._discoverHostMarkerGroup()
            self._discoverHostProjectionGroup()
            self._buildEmbeddedMapFields()
            self._discoverDataMarkerGroup()
            self._buildDataGroups()

    def getContext(self) -> Context:
        return self._context

    def getHostRegion(self) -> Region:
        """
        Get region where the host scaffold is loaded.
        :return: Zinc Region.
        """
        return self._hostRegion

    def getHostMesh(self) -> Mesh:
        """
        Get host region mesh of highest dimension.
        Only call after load().
        :return: Zinc Mesh.
        """
        return self._hostMesh

    def getHostBoundaryMesh(self) -> Mesh:
        """
        Get boundary mesh from host region, dimension one less than host mesh.
        Only call after load().
        :return: Zinc Mesh.
        """
        return self._hostBoundaryMesh

    def getFittedGroup(self) -> FieldGroup:
        """
        Get group containing elements over which fitted coordinates field is defined.
        Only call after load().
        :return: Zinc FieldGroup.
        """
        return self._fittedGroup

    def getFittedMeshGroup(self) -> MeshGroup:
        """
        Get mesh from host region over which fitted coordinates are defined, in which data coordintes are found.
        Can be a subset of host mesh.
        Only call after load().
        :return: Zinc Mesh.
        """
        return self._fittedMeshGroup

    def getFittedBoundaryGroup(self) -> FieldGroup:
        """
        Get group containing boundary of elements over which fitted coordinates field is defined.
        Only call after load().
        :return: Zinc FieldGroup.
        """
        return self._fittedBoundaryGroup

    def getFittedBoundaryMeshGroup(self) -> MeshGroup:
        """
        Get boundary mesh from host region over which fitted coordinates are defined, and data coordinates found.
        Only call after load().
        :return: Zinc Mesh.
        """
        return self._fittedBoundaryMeshGroup

    def getDataRegion(self) -> Region:
        """
        Get the child data region where the embedded data is loaded.
        :return: Zinc Region
        """
        return self._dataRegion

    def getFittedCoordinatesField(self):
        """
        Get the field on the host/scaffold region giving the fitted coordinates the data coordinates are
        relative to.
        """
        return self._fittedCoordinatesField

    def setFittedCoordinatesField(self, fittedCoordinatesField: FieldFiniteElement):
        """
        Set the field on the host/scaffold region giving the fitted coordinates the data coordinates are
        relative to.
        :param fittedCoordinatesField: Fitted coordinates field defined on the host.
        :return: True if field changed, otherwise False.
        """
        if fittedCoordinatesField == self._fittedCoordinatesField:
            return False
        finiteElementField = fittedCoordinatesField.castFiniteElement() if fittedCoordinatesField else None
        assert ((fittedCoordinatesField is not None) and
            (fittedCoordinatesField.getFieldmodule().getRegion() == self._hostRegion) and
            finiteElementField.isValid() and (fittedCoordinatesField.getNumberOfComponents() <= 3))
        self._fittedCoordinatesField = finiteElementField
        self._fittedCoordinatesFieldName = fittedCoordinatesField.getName()
        self._needGenerateOutput = True
        return True

    def getMaterialCoordinatesField(self):
        """
        Get the field on the host/scaffold region giving the material coordinates embedded locations need to supply.
        """
        return self._materialCoordinatesField

    def setMaterialCoordinatesField(self, materialCoordinatesField: FieldFiniteElement):
        """
        Set the field on the host/scaffold region giving the material coordinates embedded locations need to supply.
        :param materialCoordinatesField: Material coordinates field defined on the host.
        :return: True if field changed, otherwise False.
        """
        if materialCoordinatesField == self._materialCoordinatesField:
            return False
        finiteElementField = materialCoordinatesField.castFiniteElement() if materialCoordinatesField else None
        assert ((materialCoordinatesField is not None) and
            (materialCoordinatesField.getFieldmodule().getRegion() == self._hostRegion) and
            finiteElementField.isValid() and (materialCoordinatesField.getNumberOfComponents() <= 3))
        self._materialCoordinatesField = finiteElementField
        self._materialCoordinatesFieldName = materialCoordinatesField.getName()
        self._needGenerateOutput = True
        return True

    def getHostMarkerGroup(self):
        return self._hostMarkerGroup

    def setHostMarkerGroup(self, hostMarkerGroup: FieldGroup):
        """
        Set the marker group in the host region for UI visualisation.
        :param hostMarkerGroup: Marker group in host region, or None.
        :return: True if group changed, otherwise False.
        """
        if hostMarkerGroup == self._hostMarkerGroup:
            return False
        assert (hostMarkerGroup is None) or (hostMarkerGroup.castGroup().isValid() and
               (hostMarkerGroup.getFieldmodule().getRegion() == self._hostRegion))
        self._hostMarkerGroup = None
        self._hostMarkerGroupName = None
        self._hostMarkerLocationField = None
        self._hostMarkerNameField = None
        if not hostMarkerGroup:
            return True
        self._hostMarkerGroup = hostMarkerGroup.castGroup()
        self._hostMarkerGroupName = self._hostMarkerGroup.getName()
        hostFieldmodule = self._hostRegion.getFieldmodule()
        nodes = hostFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodesetGroup = self._hostMarkerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        if markerNodesetGroup.isValid():
            node = markerNodesetGroup.createNodeiterator().next()
            if node.isValid():
                fieldcache = hostFieldmodule.createFieldcache()
                fieldcache.setNode(node)
                fielditer = hostFieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._hostMarkerLocationField) and field.castStoredMeshLocation().isValid():
                            self._hostMarkerLocationField = field
                        elif (not self._hostMarkerNameField) and field.castStoredString().isValid():
                            self._hostMarkerNameField = field
                    field = fielditer.next()
        return True

    def getHostMarkerCoordinatesField(self, modelCoordinatesField: Field):
        """
        Get field returning value of modelCoordinatesField for the marker embedded locations.
        Can be modelCoordinatesField itself if defined on markers.
        :param modelCoordinatesField: Coordintes field
        :return: Field if valid, otherwise None.
        """
        if not (self._hostMarkerGroup and modelCoordinatesField and
                (modelCoordinatesField.getFieldmodule().getRegion() == self._hostRegion)):
            return None
        hostFieldmodule = self._hostRegion.getFieldmodule()
        # return field directly if it is defined on markers
        nodes = hostFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodesetGroup = self._hostMarkerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        fieldcache = None
        if markerNodesetGroup.isValid():
            node = markerNodesetGroup.createNodeiterator().next()
            if node.isValid():
                fieldcache = hostFieldmodule.createFieldcache()
                fieldcache.setNode(node)
                if modelCoordinatesField.isDefinedAtLocation(fieldcache):
                    return modelCoordinatesField
        if self._hostMarkerLocationField:
            return hostFieldmodule.createFieldEmbedded(modelCoordinatesField, self._hostMarkerLocationField)
        if (fieldcache) and self._materialCoordinatesField.isDefinedAtLocation(fieldcache):
            with ChangeManager(hostFieldmodule):
                hostLocation = hostFieldmodule.createFieldFindMeshLocation(
                    self._materialCoordinatesField, self._materialCoordinatesField, self._hostMesh)
                return hostFieldmodule.createFieldEmbedded(modelCoordinatesField, hostLocation)
        return None

    def getHostMarkerNameField(self):
        """
        :return: Field if valid, otherwise None.
        """
        return self._hostMarkerNameField

    def getHostProjectionGroup(self):
        return self._hostProjectionGroup

    def setHostProjectionGroup(self, hostProjectionGroup: FieldGroup):
        """
        Set the group in the host region for limiting embedded locations to be within.
        :param hostProjectionGroup: Projection group in host region, or None.
        :return: True if group changed, otherwise False.
        """
        if hostProjectionGroup == self._hostProjectionGroup:
            return False
        assert (hostProjectionGroup is None) or (hostProjectionGroup.castGroup().isValid() and
               (hostProjectionGroup.getFieldmodule().getRegion() == self._hostRegion))
        self._hostProjectionMeshGroup = None
        if hostProjectionGroup:
            self._hostProjectionGroup = hostProjectionGroup.castGroup()
            self._hostProjectionGroupName = self._hostProjectionGroup.getName()
            # get highest dimension non-empty mesh group
            hostFieldmodule = self._hostRegion.getFieldmodule()
            with ChangeManager(hostFieldmodule):
                for dimension in range(3, 0, -1):
                    mesh = hostFieldmodule.findMeshByDimension(dimension)
                    elementGroupField = self._hostProjectionGroup.getFieldElementGroup(mesh)
                    if elementGroupField.isValid():
                        meshGroup = elementGroupField.getMeshGroup()
                        if meshGroup.getSize() > 0:
                            # intersect with self._fittedGroup:
                            self._hostProjectionMeshGroup = hostFieldmodule.createFieldElementGroup(mesh).getMeshGroup()
                            self._hostProjectionMeshGroup.addElementsConditional(
                                hostFieldmodule.createFieldAnd(self._fittedGroup, self._hostProjectionGroup))
                            if self._hostProjectionMeshGroup.getSize() == 0:
                                self._hostProjectionMeshGroup = None
                            else:
                                if self._hostProjectionMeshGroup.getSize() == meshGroup.getSize():
                                    self._hostProjectionMeshGroup = meshGroup  # meshGroup was already intersection
                                break;
        else:
            self._hostProjectionGroup = None
            self._hostProjectionGroupName = None
        self._buildEmbeddedMapFields()
        self._needGenerateOutput = True
        return True

    def getDataCoordinatesField(self) -> FieldFiniteElement:
        """
        Get the field on the data region giving the coordinates to find embedded locations from.
        """
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField: FieldFiniteElement):
        """
        Set the field on the data region giving the coordinates to find embedded locations from.
        :param dataCoordinatesField: Data coordinates field defined on the data region.
        :return: True if field changed, otherwise False.
        """
        if dataCoordinatesField == self._dataCoordinatesField:
            return False
        finiteElementField = dataCoordinatesField.castFiniteElement() if dataCoordinatesField else None
        assert ((dataCoordinatesField is not None) and
            (dataCoordinatesField.getFieldmodule().getRegion() == self._dataRegion) and
            finiteElementField.isValid() and (dataCoordinatesField.getNumberOfComponents() <= 3))
        self._dataCoordinatesField = finiteElementField
        self._dataCoordinatesFieldName = dataCoordinatesField.getName()
        self._needGenerateOutput = True
        return True

    def getDataMarkerGroup(self):
        return self._dataMarkerGroup

    def setDataMarkerGroup(self, dataMarkerGroup: FieldGroup):
        """
        Set the marker group from which point data is extracted via its name field. The name field and coordinates
        field are automatically discovered from the group by looking at the datapoints in it.
        :param dataMarkerGroup: Marker group in data subregion, or None. Both fiducial markers and embedded point
        data will be in this group. A stored string field on the points gives the group name.
        :return: True if group changed, otherwise False.
        """
        if dataMarkerGroup == self._dataMarkerGroup:
            return False
        assert (dataMarkerGroup is None) or (dataMarkerGroup.castGroup().isValid() and
               (dataMarkerGroup.getFieldmodule().getRegion() == self._dataRegion))
        self._dataMarkerGroup = None
        self._dataMarkerGroupName = None
        self._dataMarkerCoordinatesField = None
        self._dataMarkerNameField = None
        if not dataMarkerGroup:
            return True
        self._dataMarkerGroup = dataMarkerGroup.castGroup()
        self._dataMarkerGroupName = self._dataMarkerGroup.getName()
        self._needGenerateOutput = True
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
            print("Data marker group", self._dataMarkerGroupName, "is empty or has no coordinates or name field",
                  file=sys.stderr)
        return True

    def getDataMarkerCoordinatesField(self):
        """
        Get field giving data marker coordinates.
        :param modelCoordinatesField:
        :return: Zinc Field or None if none.
        """
        return self._dataMarkerCoordinatesField

    def getDataMarkerNameField(self):
        return self._dataMarkerNameField

    def getDiagnosticLevel(self) -> int:
        return self._diagnosticLevel

    def setDiagnosticLevel(self, diagnosticLevel: int):
        """
        :param diagnosticLevel: 0 = no diagnostic messages. 1+ = Information and warning messages.
        """
        assert diagnosticLevel >= 0
        self._diagnosticLevel = diagnosticLevel

    def getDataGroupNames(self):
        return self._groupData.keys()

    def hasDataGroup(self, groupName: str) -> bool:
        """
        Query whether group of name exists in data.
        :param groupName: Name of the group
        :return: True if group exists, otherwise False.
        """
        return groupName in self._groupData

    def getDataGroupDimension(self, groupName: str) -> int:
        """
        Get highest dimension in objects of data group of name.
        :param groupName: Name of the group
        :return: Dimension >= 0, or -1 if group not found.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._dimensionToken]
        print("DataEmbedder getDataGroupDimension: no group of name " + str(groupName), file=sys.stderr)
        return 0

    def isDataGroupEmbed(self, groupName: str) -> bool:
        """
        Query whether data will be embedded for the group.
        :param groupName: Name of the group
        :return: True if group is to be embedded, otherwise False.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._embedToken]
        print("DataEmbedder isDataGroupEmbed: no group of name " + str(groupName), file=sys.stderr)
        return False

    def setDataGroupEmbed(self, groupName: str, embed: bool):
        """
        Set whether to embed data for the group.
        Client must call generateOutput() after making changes.
        :param groupName: Name of the group to modify settings of.
        :param embed: True to embed group data, False to not embed.
        :return: True if embed state changed, otherwise False.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            if groupDict[self._embedToken] != embed:
                groupDict[self._embedToken] = embed
                self._needGenerateOutput = True
                return True
        else:
            print("DataEmbedder setDataGroupEmbed: no group of name " + str(groupName), file=sys.stderr)
        return False

    def getDataGroupSize(self, groupName: str) -> int:
        """
        Get number of objects in data group of name of its highest dimension.
        :param groupName: Name of the group
        :return: Size > 0, or -1 if group not found.
        """
        groupDict = self._groupData.get(groupName)
        if groupDict:
            return groupDict[self._sizeToken]
        print("DataEmbedder getDataGroupSize: no group of name " + str(groupName), file=sys.stderr)
        return -1

    def printLog(self):
        loggerMessageCount = self._logger.getNumberOfMessages()
        if loggerMessageCount > 0:
            for i in range(1, loggerMessageCount + 1):
                print(self._logger.getMessageTypeAtIndex(i), self._logger.getMessageTextAtIndex(i))
            self._logger.removeAllMessages()

    def generateOutput(self) -> Region:
        """
        Generate embedded data from the groups with their embed flag set.
        :return: Zinc Region containing embedded output data. A new output data region is regenerated as needed.
        """
        if not self._needGenerateOutput:
            return self._outputDataRegion
        if self._outputDataRegion:
            self._hostRegion.removeChild(self._outputDataRegion)
        self._outputDataRegion = self._hostRegion.createChild("output")
        result = self._outputDataRegion.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file into output" + str(self._zincDataFileName)
        outputDataFieldmodule = self._outputDataRegion.getFieldmodule()
        outputMesh = [None]
        for dimension in range(1, 4):
            outputMesh.append(outputDataFieldmodule.findMeshByDimension(dimension))
        outputNodes = outputDataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        outputDatapoints = outputDataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        with ChangeManager(outputDataFieldmodule):
            # make a group containing all the objects we want to keep
            embedGroup = outputDataFieldmodule.createFieldGroup()
            embedMeshGroup = [None]
            for dimension in range(1, 4):
                embedMeshGroup.append(embedGroup.createFieldElementGroup(outputMesh[dimension]).getMeshGroup())
            embedNodeGroup = embedGroup.createFieldNodeGroup(outputNodes).getNodesetGroup()
            embedDataGroup = embedGroup.createFieldNodeGroup(outputDatapoints).getNodesetGroup()
            outputDataMarkerGroup = None
            outputDataMarkerNameField = None
            if self._dataMarkerGroup:
                outputDataMarkerGroup = outputDataFieldmodule.findFieldByName(self._dataMarkerGroupName).castGroup()
                outputDataMarkerNameField = outputDataFieldmodule.findFieldByName(
                    self._dataMarkerNameField.getName()) if self._dataMarkerNameField else None
            for groupName, groupDict in self._groupData.items():
                group = outputDataFieldmodule.findFieldByName(groupName).castGroup()
                if groupDict[self._embedToken]:
                    groupDimension = groupDict[self._dimensionToken]
                    if group.isValid():
                        for dimension in range(groupDimension, 0, -1):
                            embedMeshGroup[dimension].addElementsConditional(group)
                        embedNodeGroup.addNodesConditional(group)
                    else:
                        if not (outputDataMarkerGroup and outputDataMarkerNameField):
                            if self._diagnosticLevel > 0:
                                print("Missing data marker group or name fields for group " + groupName,
                                      file=sys.stderr)
                            continue
                        # marker points: make zinc group of datapoints with the same name
                        group = outputDataFieldmodule.createFieldGroup()
                        if group.setName(groupName) != RESULT_OK:
                            print("Failed to set marker group name " + groupName + "; skipping", file=sys.stderr)
                            del group
                            continue
                        group.setManaged(True)
                        dataGroup = group.createFieldNodeGroup(outputDatapoints).getNodesetGroup()
                        fieldcache = outputDataFieldmodule.createFieldcache()
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
            notEmbedGroup = outputDataFieldmodule.createFieldNot(embedGroup)
            for dimension in range(1, 4):
                outputMesh[dimension].destroyElementsConditional(notEmbedGroup)
            outputNodes.destroyNodesConditional(notEmbedGroup)
            outputDatapoints.destroyNodesConditional(notEmbedGroup)
            del notEmbedGroup
            del embedMeshGroup
            del embedNodeGroup
            del embedDataGroup
            # define material coordinates identically to data coordinates by writing to a temporary EX format buffer
            # handle case where marker coordinates is a different field
            outputDataCoordinatesFieldNames = [self._dataCoordinatesFieldName]
            if self._dataMarkerCoordinatesField and \
                    (not (self._dataMarkerCoordinatesField == self._dataCoordinatesField)):
                outputDataCoordinatesFieldNames.append(self._dataMarkerCoordinatesField.getName())
            # handle host material coordinates field name already being in use in data,
            # usually because it's also called "coordinates" --> prefix with "material " and warn
            outputDataMaterialCoordinatesFieldName = self._materialCoordinatesFieldName
            for outputDataCoordinatesFieldName in outputDataCoordinatesFieldNames:
                if outputDataCoordinatesFieldName == outputDataMaterialCoordinatesFieldName:
                    outputDataMaterialCoordinatesFieldName = get_unique_field_name(
                        outputDataFieldmodule, "material " + outputDataMaterialCoordinatesFieldName)
                    print("Warning: Data already has field '" + self._materialCoordinatesFieldName +
                        "' defined on it. Defining material coordinates on it with name '" +
                        outputDataMaterialCoordinatesFieldName + "' instead.", file=sys.stderr)
                    break
            buffers = []
            for outputDataCoordinatesFieldName in outputDataCoordinatesFieldNames:
                outputDataCoordinatesField = outputDataFieldmodule.findFieldByName(outputDataCoordinatesFieldName)
                # temporarily rename field for output
                outputDataCoordinatesField.setName(outputDataMaterialCoordinatesFieldName)
                sir = self._outputDataRegion.createStreaminformationRegion()
                srm = sir.createStreamresourceMemory()
                sir.setResourceFieldNames(srm, [outputDataMaterialCoordinatesFieldName])
                if outputDataCoordinatesFieldName != outputDataCoordinatesFieldNames[0]:
                    # markers should only be datapoints
                    sir.setResourceDomainTypes(srm, Field.DOMAIN_TYPE_DATAPOINTS)
                # workaround for Zinc writing empty groups when no members have above field defined on them
                sir.setResourceGroupName(srm, embedGroup.getName())
                self._outputDataRegion.write(sir)
                result, buffer = srm.getBuffer()
                assert result == RESULT_OK, "Failed to write data coordinates to memory"
                # restore field name before reading back in the cloned field
                outputDataCoordinatesField.setName(outputDataCoordinatesFieldName)
                buffers.append(buffer)
            sir = self._outputDataRegion.createStreaminformationRegion()
            for buffer in buffers:
                sir.createStreamresourceMemoryBuffer(buffer)
            result = self._outputDataRegion.read(sir)
            if result != RESULT_OK:
                self.printLog()
                assert False, "Failed to define output material coordinates from memory buffer"
            self._outputDataMaterialCoordinatesField = \
                outputDataFieldmodule.findFieldByName(outputDataMaterialCoordinatesFieldName).castFiniteElement()
            del embedGroup
            if self._outputDataMaterialCoordinatesField.isValid():
                # now do the embedding: evaluate material coordinates in output region
                for outputDataCoordinatesFieldName in outputDataCoordinatesFieldNames:
                    field = outputDataFieldmodule.findFieldByName(outputDataCoordinatesFieldName).castFiniteElement()
                    applyField = outputDataFieldmodule.createFieldApply(self._hostFindMaterialCoordinatesField)
                    result = applyField.setBindArgumentSourceField(self._coordinatesArgumentField, field)
                    assert result == RESULT_OK, "Failed to set bind argument source field in output"
                    fieldassignment = self._outputDataMaterialCoordinatesField.createFieldassignment(applyField)
                    for nodeset in ((outputNodes, outputDatapoints)
                            if (outputDataCoordinatesFieldName == outputDataCoordinatesFieldNames[0])
                            else (outputDatapoints,)):
                        fieldassignment.setNodeset(nodeset)
                        result = fieldassignment.assign()
                        assert result in (RESULT_OK, RESULT_WARNING_PART_DONE, RESULT_ERROR_NOT_FOUND), \
                            "Failed to assign material coordinates"
            else:
                if self._diagnosticLevel > 0:
                    print("No embedded data / failed to define output material coordinates field", file=sys.stderr)

        self._needGenerateOutput = False
        return self._outputDataRegion

    def getOutputDataMaterialCoordinatesField(self) -> FieldFiniteElement:
        return self._outputDataMaterialCoordinatesField

    def getOutputDataHostCoordinatesField(self, hostCoordinatesField: Field):
        """
        Get a field giving host coordinates from data material coordinates. For UI/visualisation use.
        :param hostCoordinatesField: Field to evaluate on host.
        :return: Field giving host coordinates on data at matching material coordinates, or an invalid field if none.
        """
        if not hostCoordinatesField:
            return Field()
        assert hostCoordinatesField.getFieldmodule().getRegion() == self._hostRegion
        hostFieldmodule = self._hostRegion.getFieldmodule()
        with ChangeManager(hostFieldmodule):
            findMaterialMeshLocationField = hostFieldmodule.createFieldFindMeshLocation(
                self._coordinatesArgumentField, self._materialCoordinatesField, self._hostMesh)
            # following give same result as when not set, only much faster or just faster, respectively
            searchMesh = self._hostProjectionMeshGroup if self._hostProjectionMeshGroup else self._fittedMeshGroup
            findMaterialMeshLocationField.setSearchMesh(searchMesh)
            # use SEARCH_MODE_NEAREST as derivatives or minor errors may give locations outside host mesh
            findMaterialMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
            embeddedHostCoordinatesField =\
                hostFieldmodule.createFieldEmbedded(hostCoordinatesField, findMaterialMeshLocationField)
            outputDataFieldmodule = self._outputDataRegion.getFieldmodule()
            with ChangeManager(outputDataFieldmodule):
                outputDataHostCoordinatesField = outputDataFieldmodule.createFieldApply(embeddedHostCoordinatesField)
                result = outputDataHostCoordinatesField.setBindArgumentSourceField(
                    self._coordinatesArgumentField, self._outputDataMaterialCoordinatesField)
                assert result == RESULT_OK, "Failed to set bind argument source field in output"
        return outputDataHostCoordinatesField
