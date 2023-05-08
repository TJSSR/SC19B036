#**************************************************************************************************#
# Importing Necessary Modules
#**************************************************************************************************#
import numpy as np
from random import random
import csv

# ABAQUS
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

#**************************************************************************************************#
# Master Function Definition
#**************************************************************************************************#
def Master_Function(th, tv, b, l, r1, r2, LRVE, BRVE, AR, TR, Vf, sheetsize, gridspace, 
                    rhop, Ep, vp, rhom, Em, vm, seedVal, initincval, maxincval, 
                    minincval, maxnumincval, timep, xdisp, ydisp, cpunum, gpunum, fail_p,
                      norm_fail_m, shear_fail_m):
    
    # Defining Module Functions
    def Part_Module(th, tv, b, l, r1, r2, LRVE, BRVE, sheetsize, gridspace):
        mod = mdb.models[modelname]
        mod.ConstrainedSketch(name='__profile__', sheetSize=sheetsize, ).setPrimaryObject(option=STANDALONE)
        s = mod.sketches['__profile__']
        s.rectangle(point1=(0.0, 0.0), point2=(LRVE, BRVE))
        session.viewports['Viewport: 1'].view.fitView()
        mod.Part(name=partname, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        mod.parts[partname].BaseShell(sketch=mod.sketches['__profile__'])
        mod.ConstrainedSketch(name='__profile__', sheetSize=sheetsize).unsetPrimaryObject()
        
        # Generating Partitions
        session.viewports['Viewport: 1'].setValues(displayedObject=mod.parts[partname])
        p = mod.parts[partname]
        f = p.faces
        RVEcent = (0.5*LRVE ,0.5*BRVE, 0.0)
        t = p.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=RVEcent)
        # In the above line, the coordinates set as the origin are defined with respect to the origin which was used to when sketching the part, which is
        # the bottom left corner.

        s = mod.ConstrainedSketch(name='__profile__', sheetSize=sheetsize, gridSpacing=gridspace, transform=t)
        s.setPrimaryObject(option=SUPERIMPOSE)
        p = mod.parts[partname]
        p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)

        # Note while specifying the necessary curves/shapes for partition sketches, give coordinates defined by the 'RVEcent' (translation of origin)
        s.rectangle(point1=(0-RVEcent[0], BRVE-RVEcent[1]), point2=(LRVE-RVEcent[0], BRVE-0.5*th-RVEcent[1]))   # upper horizontal matrix layer
        s.rectangle(point1=(0.5*tv-RVEcent[0], BRVE-0.5*th-RVEcent[1]), point2=(LRVE-0.5*tv-RVEcent[0], BRVE-0.5*th-b-RVEcent[1]))  # upper platelet
        s.rectangle(point1=(0-RVEcent[0], b+1.5*th-RVEcent[1]), point2=(LRVE-RVEcent[0], b+0.5*th-RVEcent[1]))  # middle horizontal matrix layer
        s.rectangle(point1=(0-RVEcent[0], b+0.5*th-RVEcent[1]), point2=(r1*l-RVEcent[0], 0.5*th-RVEcent[1]))  # lower left platelet
        s.rectangle(point1=(LRVE-r2*l-RVEcent[0], 0.5*th+b-RVEcent[1]), point2=(LRVE-RVEcent[0], 0.5*th-RVEcent[1]))  # lower right platelet
        s.rectangle(point1=(0-RVEcent[0], 0.5*th-RVEcent[1]), point2=(LRVE-RVEcent[0], 0-RVEcent[1]))   # lower horizontal matrix layer
        s.rectangle(point1=(0-RVEcent[0], BRVE-0.5*th-RVEcent[1]), point2=(0.5*tv-RVEcent[0], b+1.5*th-RVEcent[1]))   # upper left vertical matrix layer
        s.rectangle(point1=(LRVE-0.5*tv-RVEcent[0], BRVE-0.5*th-RVEcent[1]), point2=(LRVE-RVEcent[0], b+1.5*th-RVEcent[1]))   # upper right vertical matrix layer
        s.rectangle(point1=(r1*l-RVEcent[0], 0.5*th+b-RVEcent[1]), point2=(r1*l+tv-RVEcent[0], 0.5*th-RVEcent[1]))   # lower middle vertical matrix layer
        
        p = mod.parts[partname]

        # Parametrically defining faces for partition based on above created rectangles. The following are the centroids of those rectangles
        # These coordinates are with respect to the origin used for creating the sketch for the part
        pt1 = (0.5*LRVE, BRVE-0.25*th, 0)
        f1 = p.faces.findAt(pt1)                    # upper horizontal matrix layer
        pt2 = (0.5*LRVE, 0.75*BRVE, 0)
        f2 = p.faces.findAt(pt2)                    # upper platelet
        pt3 = (0.5*LRVE, 0.5*BRVE, 0)
        f3 = p.faces.findAt(pt3)                    # middle horizontal matrix layer
        pt4 = (0.5*r1*l, 0.25*BRVE, 0)
        f4 = p.faces.findAt(pt4)                    # lower left platelet
        pt5 = (LRVE-0.5*r2*l, 0.25*BRVE, 0)
        f5 = p.faces.findAt(pt5)                    # lower right platelet
        pt6 = (0.5*LRVE, 0.25*th, 0)
        f6 = p.faces.findAt(pt6)                    # lower horizontal matrix layer
        pt7 = (0.5*tv, 0.75*BRVE, 0)
        f7 = p.faces.findAt(pt7)                    # upper left vertical matrix layer
        pt8 = (LRVE-0.5*tv, 0.75*BRVE, 0)
        f8 = p.faces.findAt(pt8)                    # upper right vertical matrix layer
        pt9 = (0.5*LRVE, 0.25*BRVE, 0)
        f9 = p.faces.findAt(pt9)                    # lower vertical matrix layer

        
        pickedfaces = (f1, f2, f3, f4, f5, f6, f7, f8, f9, )
        p.PartitionFaceBySketch(faces=pickedfaces, sketch=s)
        s.unsetPrimaryObject()

    def Property_Module(rhop, Ep, vp, rhom, Em, vm, th, tv, l, r1, r2, LRVE, BRVE):
        mod = mdb.models[modelname]
        # Platelet Material
        mod.Material(name=matp)
        mod.materials[matp].Density(table=((rhop, ), ))
        mod.materials[matp].Elastic(type=ISOTROPIC, table=((Ep, vp), ))
        # mod.materials[matp].Plastic(table=((240,0), ))

        # Matrix Material
        mod.Material(name=matm)
        mod.materials[matm].Density(table=((rhom, ), ))
        mod.materials[matm].Elastic(type=ISOTROPIC, table=((Em, vm), ))
        # matm_pt = ExcelDataToScriptData('TilAl4V_ELI_Stress_Strain.xlsx', 'Sheet1')         # Usage of the plastic table via 'ExcelDatatoScriptData' fucntion
        # matrix_plastic_table = ((790.0, 0.0), (791.23, 0.00022), (795.05, 0.00085), (798.87, 0.00165), (802.69, 0.00265), (806.51, 0.00391), (810.33, 0.0055), (814.15, 0.0075), (817.97, 0.01003), (821.79, 0.0132), (825.62, 0.01719), (829.44, 0.02221), (833.26, 0.02851), (837.08, 0.03642), (840.9, 0.04635), (844.72, 0.05879), (848.54, 0.07437), (852.36, 0.09387), (856.18, 0.11823))
        # mod.materials[matm].Plastic(table=matrix_plastic_table)

        # Platelet Section
        p = mod.parts[partname]
        psecname = 'Platelet_Section'

        mod.HomogeneousSolidSection(material=matp, name=psecname, thickness=100)

        p1 = (round(0.5*LRVE, 6), round(0.75*BRVE, 6), 0.0) # top platelet
        p2 = (round(0.5*r1*l, 6), round(0.25*BRVE, 6), 0.0) # bottom left platelet
        p3 = (round(LRVE-0.5*r2*l, 6), round(0.25*BRVE, 6), 0.0)# upper left quarter of bottom right platelet

        p.Set(name='Platelet_Faces', faces=p.faces.findAt((p1, ), (p2, ), (p3, ), ))

        p.SectionAssignment(region=p.sets['Platelet_Faces'], sectionName=psecname, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

        # Matrix Section
        p = mod.parts[partname]
        msecname = 'Matrix_Section'

        mod.HomogeneousSolidSection(material=matm, name=msecname, thickness=100)

        p1 = (round(0.5*LRVE, 6), round(BRVE-0.25*th, 6), 0)# upper horizontal matrix layer

        p2 = (round(0.5*LRVE, 6), round(0.5*BRVE, 6), 0)# middle horizontal matrix layer

        p3 = (round(0.5*LRVE, 6), round(0.25*th, 6), 0)# lower horizontal matrix layer

        p4 = (round(0.25*tv, 6), round(0.75*BRVE, 6), 0)# upper left vertical matrix layer

        p5 = (round(LRVE-0.25*tv, 6), round(0.75*BRVE, 6), 0)# upper right vertical matrix layer

        p6 = (round(r1*l+0.5*tv, 6), round(0.25*BRVE, 6), 0)# lower vertical matrix layer

        p.Set(name='Matrix_Faces', faces=p.faces.findAt((p1, ), (p2, ), (p3, ), (p4, ), (p5, ), (p6, ), ))
        p.Set(name='VI_FACES', faces=p.faces.findAt((p4, ), (p5, ), (p6, ), ))
        p.Set(name='HI_FACES', faces=p.faces.findAt((p1, ), (p2, ), (p3, ), ))
        p.SectionAssignment(region=p.sets['Matrix_Faces'], sectionName=msecname, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    def Assembly_Module():
        mod = mdb.models[modelname]
        a = mod.rootAssembly
        a.DatumCsysByDefault(CARTESIAN)
        p = mod.parts[partname]
        a.Instance(name=instancename, part=p, dependent=OFF)

    def Mesh_Module(seedVal, th, tv, b, l, r1, r2, LRVE, BRVE):
        mod = mdb.models[modelname]
        a = mod.rootAssembly
        c = a.instances[instancename]
        session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
        session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(meshTechnique=ON)

        # Creating Face Sets
        p1 = (round(0.5*LRVE, 6), round(BRVE-0.25*th, 6), 0.0)
        p2 = (round(0.25*tv, 6), round(BRVE-0.5*th-0.5*b, 6), 0.0)
        p3 = (round(LRVE-0.25*tv, 6), round(BRVE-0.5*th-0.5*b, 6), 0.0)
        p4 = (round(0.5*LRVE, 6), round(0.5*BRVE, 6), 0.0)
        p5 = (round(r1*l+0.5*tv, 6), round(0.25*BRVE, 6), 0.0)
        p6 = (round(0.5*LRVE, 6), round(0.25*th, 6), 0.)

        p7 = (round(0.5*LRVE, 6), round(0.75*BRVE, 6), 0.0)
        p8 = (round(0.5*r1*l, 6), round(0.25*BRVE, 6), 0.0)
        p9 = (round(LRVE-0.5*r2*l, 6), round(0.25*BRVE, 6), 0.0)
        
        a.seedPartInstance(regions=(c, ), size=seedVal, deviationFactor=0.1, minSizeFactor=0.1)

        # Setting Element Type and Assigning Mesh Controls
        elemType1 = ElemType(elemCode=CPE4, elemLibrary=STANDARD)
        elemType2 = ElemType(elemCode=CPE3, elemLibrary=STANDARD)

        a.setElementType(elemTypes=(elemType1, elemType2), regions=(c.faces.findAt((p1, ), (p2, ), (p3, ), (p4, ), (p5, ), (p6, ), (p7, ), (p8, ), (p9, ), ), ))
        a.setMeshControls(elemShape=QUAD, regions=c.faces.findAt((p1, ), (p2, ), (p3, ), (p4, ), (p5, ), (p6, ), (p7, ), (p8, ), (p9, ), ), technique=STRUCTURED)
        a.generateMesh(regions=(c, ))

    def Step_Module(initincval, maxincval, minincval, maxnumincval, timep):
        mod = mdb.models[modelname]
        mod.StaticStep(initialInc=initincval, maxInc=maxincval, minInc=minincval, maxNumInc=maxnumincval, name=stepname, previous='Initial', timePeriod=timep)

    def Eqn_Constraints(th, LRVE, BRVE):
        mod = mdb.models[modelname]
        a = mod.rootAssembly
        p = a.instances[instancename]
        L1 = p.edges.findAt((0.0, round(0.25*th, 6), 0.0))
        L2 = p.edges.findAt((0.0, round(0.25*BRVE, 6), 0.0))
        L3 = p.edges.findAt((0.0, round(0.5*BRVE, 6), 0.0))
        L4 = p.edges.findAt((0.0, round(0.75*BRVE, 6), 0.0))
        L5 = p.edges.findAt((0.0, round(BRVE-0.25*th, 6), 0.0))

        R1 = p.edges.findAt((round(LRVE, 6), round(0.25*th, 6), 0.0))
        R2 = p.edges.findAt((round(LRVE, 6), round(0.25*BRVE, 6), 0.0))
        R3 = p.edges.findAt((round(LRVE, 6), round(0.5*BRVE, 6), 0.0))
        R4 = p.edges.findAt((round(LRVE, 6), round(0.75*BRVE, 6), 0.0))
        R5 = p.edges.findAt((round(LRVE, 6), round(BRVE-0.25*th, 6), 0.0))

        U = p.edges.findAt((round(0.5*LRVE, 6), round(BRVE, 6), 0.0))
        D = p.edges.findAt((round(0.5*LRVE, 6), 0.0, 0.0))

        q1 = U.index
        q2 = D.index

        q31 = R1.index
        q32 = R2.index
        q33 = R3.index
        q34 = R4.index
        q35 = R5.index

        q41 = L1.index
        q42 = L2.index
        q43 = L3.index
        q44 = L4.index
        q45 = L5.index

        EdUp = p.edges[q1:q1+1]
        a.Set(edges=EdUp, name='Up')
        Upnodes = a.sets['Up'].nodes
        EdDo = p.edges[q2:q2+1]
        a.Set(edges=EdDo, name='Down')
        Downnodes = a.sets['Down'].nodes

        EdRe1 = p.edges[q31:q31+1]
        EdRe2 = p.edges[q32:q32+1]
        EdRe3 = p.edges[q33:q33+1]
        EdRe4 = p.edges[q34:q34+1]
        EdRe5 = p.edges[q35:q35+1]

        a.Set(edges=EdRe1, name='Right-1')
        a.Set(edges=EdRe2, name='Right-2')
        a.Set(edges=EdRe3, name='Right-3')
        a.Set(edges=EdRe4, name='Right-4')
        a.Set(edges=EdRe5, name='Right-5')

        R1nodes = a.sets['Right-1'].nodes
        R2nodes = a.sets['Right-2'].nodes
        R3nodes = a.sets['Right-3'].nodes
        R4nodes = a.sets['Right-4'].nodes
        R5nodes = a.sets['Right-5'].nodes

        EdLe1 = p.edges[q41:q41+1]
        EdLe2 = p.edges[q42:q42+1]
        EdLe3 = p.edges[q43:q43+1]
        EdLe4 = p.edges[q44:q44+1]
        EdLe5 = p.edges[q45:q45+1]

        a.Set(edges=EdLe1, name='Left-1')
        a.Set(edges=EdLe2, name='Left-2')
        a.Set(edges=EdLe3, name='Left-3')
        a.Set(edges=EdLe4, name='Left-4')
        a.Set(edges=EdLe5, name='Left-5')

        L1nodes = a.sets['Left-1'].nodes
        L2nodes = a.sets['Left-2'].nodes
        L3nodes = a.sets['Left-3'].nodes
        L4nodes = a.sets['Left-4'].nodes
        L5nodes = a.sets['Left-5'].nodes

        Upcoord = []
        Downcoord = []

        L1coord = []
        L2coord = []
        L3coord = []
        L4coord = []
        L5coord = []

        R1coord = []
        R2coord = []
        R3coord = []
        R4coord = []
        R5coord = []

        for node in Upnodes:
            Upcoord = Upcoord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in Downnodes:
            Downcoord = Downcoord + [[node.coordinates[0], node.coordinates[1], node.label]]

        for node in L1nodes:
            L1coord = L1coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in L2nodes:
            L2coord = L2coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in L3nodes:
            L3coord = L3coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in L4nodes:
            L4coord = L4coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in L5nodes:
            L5coord = L5coord + [[node.coordinates[0], node.coordinates[1], node.label]]

        for node in R1nodes:
            R1coord = R1coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in R2nodes:
            R2coord = R2coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in R3nodes:
            R3coord = R3coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in R4nodes:
            R4coord = R4coord + [[node.coordinates[0], node.coordinates[1], node.label]]
        for node in R5nodes:
            R5coord = R5coord + [[node.coordinates[0], node.coordinates[1], node.label]]

        Upcoord.sort()
        Downcoord.sort()

        L1coord.sort()
        L2coord.sort()
        L3coord.sort()
        L4coord.sort()
        L5coord.sort()

        R1coord.sort()
        R2coord.sort()
        R3coord.sort()
        R4coord.sort()
        R5coord.sort()

        NodeTol = seedVal/200

        # Up and Down
        Num = len(Upcoord)
        for i in range(0,Num):
            if (abs(Upcoord[i][0]-Downcoord[i][0])<NodeTol):
                Nlabel = Upcoord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='UpNode-'+str(i))
                Nlabel = Downcoord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='DownNode-'+str(i))

        # Left and Right - corners of each edge overlap; while creating the constraint equations ignore the node with leas index for each edge along the left and right edges
        Num = len(L1coord)
        for i in range(0, Num):
            if abs(R1coord[i][1]-L1coord[i][1]<NodeTol):
                Nlabel = R1coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='R1Node-'+str(i))
                Nlabel = L1coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='L1Node-'+str(i))

        Num = len(L2coord)
        for i in range(0, Num):
            if abs(R2coord[i][1]-L2coord[i][1]<NodeTol):
                Nlabel = R2coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='R2Node-'+str(i))
                Nlabel = L2coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='L2Node-'+str(i))

        Num = len(L3coord)
        for i in range(0, Num):
            if abs(R3coord[i][1]-L3coord[i][1]<NodeTol):
                Nlabel = R3coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='R3Node-'+str(i))
                Nlabel = L3coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='L3Node-'+str(i))

        Num = len(L4coord)
        for i in range(0, Num):
            if abs(R4coord[i][1]-L4coord[i][1]<NodeTol):
                Nlabel = R4coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='R4Node-'+str(i))
                Nlabel = L4coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='L4Node-'+str(i))

        Num = len(L5coord)
        for i in range(0, Num):
            if abs(R5coord[i][1]-L5coord[i][1]<NodeTol):
                Nlabel = R5coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='R5Node-'+str(i))
                Nlabel = L5coord[i][2]
                a.Set(nodes=p.nodes[Nlabel-1:Nlabel], name='L5Node-'+str(i))


        # Constraint Equations b/w left and right edges
        for i in range(1, len(R1coord)):
            mod.Equation(name='Eqn-LR1-X-'+str(i), terms=((-1.0, 'L1Node-'+str(i), 1), (1.0, 'R1Node-'+str(i), 1), (-1.0, 'R1Node-0', 1)))
        for i in range(1, len(R1coord)):
            mod.Equation(name='Eqn-LR1-Y-'+str(i), terms=((1.0, 'L1Node-'+str(i), 2), (-1.0, 'R1Node-'+str(i), 2)))

        for i in range(1, len(R2coord)):
            mod.Equation(name='Eqn-LR2-X-'+str(i), terms=((-1.0, 'L2Node-'+str(i), 1), (1.0, 'R2Node-'+str(i), 1), (-1.0, 'R1Node-0', 1)))
        for i in range(1, len(R2coord)):
            mod.Equation(name='Eqn-LR2-Y-'+str(i), terms=((1.0, 'L2Node-'+str(i), 2), (-1.0, 'R2Node-'+str(i), 2)))

        for i in range(1, len(R3coord)):
            mod.Equation(name='Eqn-LR3-X-'+str(i), terms=((-1.0, 'L3Node-'+str(i), 1), (1.0, 'R3Node-'+str(i), 1), (-1.0, 'R1Node-0', 1)))
        for i in range(1, len(R3coord)):
            mod.Equation(name='Eqn-LR3-Y-'+str(i), terms=((1.0, 'L3Node-'+str(i), 2), (-1.0, 'R3Node-'+str(i), 2)))

        for i in range(1, len(R4coord)):
            mod.Equation(name='Eqn-LR4-X-'+str(i), terms=((-1.0, 'L4Node-'+str(i), 1), (1.0, 'R4Node-'+str(i), 1), (-1.0, 'R1Node-0', 1)))
        for i in range(1, len(R4coord)):
            mod.Equation(name='Eqn-LR4-Y-'+str(i), terms=((1.0, 'L4Node-'+str(i), 2), (-1.0, 'R4Node-'+str(i), 2)))

        for i in range(1, len(R5coord)-1):
            mod.Equation(name='Eqn-LR5-X-'+str(i), terms=((-1.0, 'L5Node-'+str(i), 1), (1.0, 'R5Node-'+str(i), 1), (-1.0, 'R1Node-0', 1)))
        for i in range(1, len(R5coord)-1):
            mod.Equation(name='Eqn-LR5-Y-'+str(i), terms=((1.0, 'L5Node-'+str(i), 2), (-1.0, 'R5Node-'+str(i), 2)))

        # Constraint Equations between the Up and Down edges
        for i in range(1, len(Upcoord)-1):
            mod.Equation(name='Eqn-UD-Y-'+str(i), terms=((-1.0, 'DownNode-'+str(i), 2), (1.0, 'UpNode-'+str(i), 2), (-1.0, 'UpNode-0', 2)))
        for i in range(1, len(Upcoord)-1):
            mod.Equation(name='Eqn-UD-X-'+str(i), terms=((1.0, 'DownNode-'+str(i), 1), (-1.0, 'UpNode-'+str(i), 1)))

        # Constraint Equations for the Top Right Corner Node
        mod.Equation(name='Eqn-TR-X', terms=((1.0, 'UpNode-'+str(len(Upcoord)-1), 1), (-1.0, 'DownNode-'+str(len(Upcoord)-1), 1)))  # along the X direction
        mod.Equation(name='Eqn-TR-Y', terms=((1.0, 'UpNode-'+str(len(Upcoord)-1), 2), (-1.0, 'UpNode-0', 2)))   # along Y direction

        return [Upcoord, Downcoord, L1coord, L2coord, L3coord, L4coord, L5coord]

    def Boundary_Conditions(xdisp, ydisp, LRVE, BRVE):
        mod = mdb.models[modelname]
        a = mod.rootAssembly
        v = a.instances[instancename].vertices
        ver = v.findAt((0.0, 0.0, 0.0))
        q = ver.index
        FixVer = v[q:q+1]
        region = a.Set(vertices=FixVer, name='Set-Fix')
        mod.EncastreBC(name='Fix', createStepName='Initial', region=region, localCsys=None)

        # Fix Upper Left Node along X direction
        ver = v.findAt((0.0, round(BRVE, 6), 0.0))
        q = ver.index
        MoveVer = v[q:q+1]
        region = a.Set(vertices=MoveVer, name='Y')
        mod.DisplacementBC(name='Y', createStepName=stepname, region=region, u1=0.0, u2=ydisp, ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)

        # Fix Bottom Right Node along Y direction
        ver = v.findAt((round(LRVE, 6), 0.0, 0.0))
        q = ver.index
        MoveVer = v[q:q+1]
        region = a.Set(vertices=MoveVer, name='X')
        mod.DisplacementBC(name='X', createStepName=stepname, region=region, u1=xdisp, u2=0.0, ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None) # change the u1 value to 'xdisp' for cases other than validation

    def Job_Module(cpunum, gpunum):
        mdb.Job(name=jobname, model=modelname, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, 
        numCpus=cpunum, numGPUs=gpunum, numDomains=cpunum)
        mdb.jobs[jobname].submit(consistencyChecking=OFF)
        mdb.jobs[jobname].waitForCompletion()

    def Post_Processing(BRVE, fail_p, norm_fail_m, shear_fail_m, Upcoord, Downcoord):
        # Accessing ODB
        odb = session.openOdb(name=odbpath)
        o2 = odb.rootAssembly
        o3 = o2.instances[instancename]

        session.viewports['Viewport: 1'].setValues(displayedObject=odb)
        session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
        session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S11'))
        session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM, uniformScaleFactor=0)

        S11_Plt_val = []
        S11_VI_val = []
        S12_HI_val = []

        S11_plt = odb.steps[stepname].frames[-1].fieldOutputs['S'].getSubset(region=o3.elementSets['PLATELET_FACES'])
        for s in S11_plt.values:
            S11_Plt_val.append(s.data[0])

        S11_VI = odb.steps[stepname].frames[-1].fieldOutputs['S'].getSubset(region=o3.elementSets['VI_FACES'])
        for s in S11_VI.values:
            S11_VI_val.append(s.data[0])

        S12_HI = odb.steps[stepname].frames[-1].fieldOutputs['S'].getSubset(region=o3.elementSets['HI_FACES'])
        for s in S12_HI.values:
            S12_HI_val.append(s.data[2])
        
        def average(list):
            return sum(list)/len(list)
        
        s11_plt_avg = average(S11_Plt_val)
        s11_vi_avg = average(S11_VI_val)
        S12_hi_avg = average(S12_HI_val)

        R_p = s11_plt_avg/fail_p
        R_v = s11_vi_avg/norm_fail_m
        R_h = S12_hi_avg/shear_fail_m

        # print(round(R_p, 6), round(R_v, 6), round(R_h, 6))

        session.Path(name='Middle', type=NODE_LIST, expression=(('RVE', (Downcoord[int(np.ceil(len(Downcoord)*0.5))][2], Upcoord[int(np.ceil(len(Upcoord)*0.5))][2], )), ))
        pth = session.paths['Middle']
        session.XYDataFromPath(name='XYData-Middle', path=pth, includeIntersections=True, projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_Y)

        x0 = session.xyDataObjects['XYData-Middle']
        total=0
        for i in range(0,(len(x0)-1)):
            a1=list(x0[i])
            a2=list(x0[i+1])
            c=0.5*(a2[0]-a1[0])*(a2[1]+a1[1])
            total=c+total
        S11_avg= total/BRVE
        # print(round(S11_avg, 6))

        session.odbs[odbpath].close()
        mdb.close()

        return round(R_p, 6), round(R_v, 6), round(R_h, 6), round(S11_avg, 6)
    
    def Del_Module(Upcoord, L1coord, L2coord, L3coord, L4coord, L5coord):
        mod = mdb.models[modelname]
        a1 = mod.rootAssembly

        # Deleting sets containing individual nodes
        for i in range(len(Upcoord)):
            del a1.sets['UpNode-'+str(i)]
            del a1.sets['DownNode-'+str(i)]
        for i in range(len(L1coord)):
            del a1.sets['L1Node-'+str(i)]
            del a1.sets['R1Node-'+str(i)]
        for i in range(len(L2coord)):
            del a1.sets['L2Node-'+str(i)]
            del a1.sets['R2Node-'+str(i)]
        for i in range(len(L3coord)):
            del a1.sets['L3Node-'+str(i)]
            del a1.sets['R3Node-'+str(i)]
        for i in range(len(L4coord)):
            del a1.sets['L4Node-'+str(i)]
            del a1.sets['R4Node-'+str(i)]
        for i in range(len(L5coord)):
            del a1.sets['L5Node-'+str(i)]
            del a1.sets['R5Node-'+str(i)]
        
        # Deleting all equation constraints
        for i in range(1, len(Upcoord)-1):
            del mod.constraints['Eqn-UD-X-'+str(i)]
            del mod.constraints['Eqn-UD-Y-'+str(i)]
        for i in range(1, len(L1coord)):
            del mod.constraints['Eqn-LR1-X-'+str(i)]
            del mod.constraints['Eqn-LR1-Y-'+str(i)]
        for i in range(1, len(L2coord)):
            del mod.constraints['Eqn-LR2-X-'+str(i)]
            del mod.constraints['Eqn-LR2-Y-'+str(i)]
        for i in range(1, len(L3coord)):
            del mod.constraints['Eqn-LR3-X-'+str(i)]
            del mod.constraints['Eqn-LR3-Y-'+str(i)]
        for i in range(1, len(L4coord)):
            del mod.constraints['Eqn-LR4-X-'+str(i)]
            del mod.constraints['Eqn-LR4-Y-'+str(i)]
        for i in range(1, len(L5coord)-1):
            del mod.constraints['Eqn-LR5-X-'+str(i)]
            del mod.constraints['Eqn-LR5-Y-'+str(i)]
            
        del mod.constraints['Eqn-TR-X']
        del mod.constraints['Eqn-TR-Y']

    # Calling Module Functions and storing outputs
    Part_Module(th, tv, b, l, r1, r2, LRVE, BRVE, sheetsize, gridspace)
    Property_Module(rhop, Ep, vp, rhom, Em, vm, th, tv, l, r1, r2, LRVE, BRVE)
    Assembly_Module()
    Mesh_Module(seedVal, th, tv, b, l, r1, r2, LRVE, BRVE)
    Step_Module(initincval, maxincval, minincval, maxnumincval, timep)
    Eqn_Constraints(th, LRVE, BRVE)
    Coord = Eqn_Constraints(th, LRVE, BRVE)
    Upcoord =Coord[0]
    Downcoord = Coord[1]
    L1coord = Coord[2]
    L2coord = Coord[3]
    L3coord = Coord[4]
    L4coord = Coord[5]
    L5coord = Coord[6]
    Boundary_Conditions(xdisp, ydisp, LRVE, BRVE)
    Job_Module(cpunum, gpunum)
    Del_Module(Upcoord, L1coord, L2coord, L3coord, L4coord, L5coord)
    Post_Processing(BRVE, fail_p, norm_fail_m, shear_fail_m, Upcoord, Downcoord)
    res1 = Post_Processing(BRVE, fail_p, norm_fail_m, shear_fail_m, Upcoord, Downcoord)

    return [b, AR, TR, Vf, r1, res1[0], res1[1], res1[2], res1[3]]

#**************************************************************************************************#
# Defining Random Hypercube Sampling and Parameter Ranges using it
#**************************************************************************************************#
def Random_Latin_Hypercube(num_samples, param_ranges):
    # num_samples = number of samples (an integer value only)
    # param_ranges = a list of parameter ranges, ex: for a two parameter case
    # param_ranges = [(1, 2), (3, 4)]
    
    num_var = len(param_ranges) # number of variables
    data = np.zeros((num_samples, num_var)) # np.zeros((num_rows, num_columns)). output is a matrix

    # Generating random values for each parameter within its range
    for i in range(num_var):    # iterating through columns
        for j in range(num_samples):    # iterating through rows = number of samples = number of stratifications
            data[j, i] = param_ranges[i][0] + (param_ranges[i][1]-param_ranges[i][0])*(j+random())/num_samples

    # shuffling the rows to create a more random Hypercube
    for i in range(num_var):    # iterating through the number of variables, i.e. columns
        order = np.arange(num_samples)  # arranging the the sample number i.e. the row number in an increasing order starting with zero
        np.random.shuffle(order)
        data[:, i] = data[order, i]
    return data

b_range = (5E-3, 10E-3)
AR_range = (2, 50)
TR_range = (-2, 2)
Vf_range = (0.4, 0.8)
r1_range = (0.1, 0.9)

param_ranges = [b_range, AR_range, TR_range, Vf_range, r1_range]
num_samples = 8000
InpData = Random_Latin_Hypercube(num_samples, param_ranges)
b = []
AR = []
TR = []
Vf = []
r1 = []
for i in range(num_samples):
    b.append(InpData[i, 0])
    AR.append(InpData[i, 1])
    TR.append(InpData[i, 2])
    Vf.append(InpData[i, 3])
    r1.append(InpData[i, 4])

#**************************************************************************************************#
# Defining software specific parameters
#**************************************************************************************************#
modelname = 'Model-1'
partname = 'Part'
sheetsize = 5000
gridspace = 0.5
matp = 'Platelet_Material'
matm = 'Matrix_Material'
instancename = 'RVE'
stepname = 'Step-1'
initincval = 1
maxincval = initincval
minincval = 0.1
timep = 1
maxnumincval = 1
jobname = 'Master'
cpunum = 6
gpunum = 1
odbpath='Master.odb'

#**************************************************************************************************#
# Defining Material Properties | Units: ton, mm, second, N, MPa, N-mm, ton/mm^3
#**************************************************************************************************#
fail_p = 260    # UTS of paltelet in MPa form MatWeb
norm_fail_m = 80    # UTS of matrix in MPa from MatWeb
shear_fail_m = 60   # Shear Strength of matrix in MPa from MatWeb

# Platelet - 99.5% Alumina
Ep = 375000             # Young's Modulus of Platelet in MPa
vp = 0.22               # Poisson's Ratio of Platelet
rhop = 3.89E-9           # Density of Platelet material in tonne/mm3

# Matrix - PMMA
Em = 2500             # Young's Modulus of Matrix in MPa
rhom = 1.185E-9         # Density of Matrix material in tonne/mm3
vm = 0.375              # Poisson's Ratio (0.35 - 0.4) (explicit reference - https://www.mit.edu/~6.777/matprops/pmma.htm)

#**************************************************************************************************#
# Calling Master_Function() for each set of variables
#**************************************************************************************************#
l = []          # length of platelet
th = []         # thickness of horizontally aligned matrix layer
tv = []         # thickness of vertically aligned matrix layer
LRVE = []       # Length of RVE
BRVE = []       # Breadth of RVE
r2 = []         # Complementary Offset percentage
xdisp = []
ydisp = []
seedVal = []    # Seed value for meshing
Result = [0]*9     # Container for all results

with open('5000Samples_MyPC.csv', 'w') as file:
    writer = csv.writer(file)
    for i in range(0, len(b)):
        a1=b[i]*AR[i]
        l.append(a1)

        c1 = np.e**TR[i]
        c2 = l[i]+b[i]*np.e**TR[i]
        c3 = l[i]*b[i]*(1-1/Vf[i])
        a2 = (-c2+np.sqrt(c2**2-4*c1*c3))/(2*c1)# Thickness of Horizontally aligned matrix in mm
        th.append(a2)

        a3 = th[i]*np.e**TR[i]    # Thickness of Vertically aligned matrix in mm
        tv.append(a3)

        a4 = l[i]+tv[i]
        LRVE.append(a4)

        a5 = 2*(b[i]+th[i])
        BRVE.append(a5)

        a6 = 1-r1[i]
        r2.append(a6)

        a7 = 0.001*LRVE[i]
        xdisp.append(a7)

        a8 = 0.001*BRVE[i]
        ydisp.append(a8)

        a9 = b[i]/(2**3)
        seedVal.append(a9)

        R = Master_Function(th[i], tv[i], b[i], l[i], r1[i], r2[i], LRVE[i], BRVE[i], AR[i], TR[i], Vf[i], sheetsize, gridspace, 
                        rhop, Ep, vp, rhom, Em, vm, seedVal[i], initincval, maxincval, 
                        minincval, maxnumincval, timep, xdisp[i], ydisp[i], cpunum, gpunum, fail_p,
                        norm_fail_m, shear_fail_m)

        Result = [R[j] for j in range(9)]
        writer.writerow(Result)
#**************************************************************************************************#
# Exporting Result to CSV file as DataSet for Machine Learning
#**************************************************************************************************#