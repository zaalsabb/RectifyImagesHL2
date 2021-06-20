import cv2
import numpy as np
import vtk
from pyoctree import pyoctree as ot
from scipy.optimize import leastsq

norm = np.linalg.norm

def ProjectToImage(projectionMatrix, pos):

    pos = pos.T
    pos_ = np.vstack([pos, np.ones((1, pos.shape[1]))])

    uv = np.dot(projectionMatrix, pos_)
    uv = uv[:, uv[-1, :] > 0]
    uv = (uv[:-1, :]/uv[-1, :]).T

    return uv

def ProjectToWorld(projectionMatrix, uv):

    uv1 = np.vstack([uv[0,:], uv[1,:], np.ones((1, uv.shape[1]))])
    pinvProjectionMatrix = np.linalg.pinv(projectionMatrix)

    rayList = np.dot(pinvProjectionMatrix, uv1)
    rayList[-1,rayList[-1,:]==0] = 1
    rayList = rayList[:-1,:]/rayList[-1,:]

    return rayList

def ProjectToWorld(projectionMatrix, uv):

    uv1 = np.vstack([uv[0,:], uv[1,:], np.ones((1, uv.shape[1]))])
    pinvProjectionMatrix = np.linalg.pinv(projectionMatrix)

    rayList = np.dot(pinvProjectionMatrix, uv1)
    rayList[-1,rayList[-1,:]==0] = 1
    rayList = rayList[:-1,:]/rayList[-1,:]

    return rayList

def plane_leastsq(points_3d):

    xs = points_3d[:,0]
    ys = points_3d[:,1]
    zs = points_3d[:,2]

    if len(xs) == 3:
        xs = np.insert(xs, 0, np.mean(xs))
        ys = np.insert(ys, 0, np.mean(ys))
        zs = np.insert(zs, 0, np.mean(zs))

    p0 = [1, 1, 1, 1]
    sol = leastsq(plane_error, p0, args=(xs, ys, zs))[0]

    A, B, C, _ = sol
    nz = np.array([A, B, C])/np.linalg.norm([A, B, C])    
    x0, y0, z0 = np.mean([xs, ys, zs], axis=1)

    return nz, np.array([x0,y0,z0])

def plane_error(p, xs, ys, zs):
    A = p[0]
    B = p[1]
    C = p[2]
    D = p[3]
    return abs(A*xs + B*ys + C*zs + D) / np.sqrt(A**2 + B**2 + C**2)

def back_project_to_mesh(img_coords,projectionMatrix,C,mesh_file):
    # Read stl file using vtk
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_file)
    reader.MergingOn()
    reader.Update()
    stl = reader.GetOutput()

    # Extract polygon info from stl
    # 1. Get array of point coordinates
    numPoints   = stl.GetNumberOfPoints()
    pointCoords = np.zeros((numPoints,3),dtype=float)
    for i in range(numPoints):
        pointCoords[i,:] = stl.GetPoint(i)
        
    # 2. Get polygon connectivity
    numPolys     = stl.GetNumberOfCells()
    connectivity = np.zeros((numPolys,3),dtype=np.int32)
    for i in range(numPolys):
        atri = stl.GetCell(i)
        ids = atri.GetPointIds()
        for j in range(3):
            connectivity[i,j] = ids.GetId(j)    

    # Create octree structure containing stl poly mesh
    tree = ot.PyOctree(pointCoords,connectivity)
    points_list = []

    for p in img_coords:
        p = p.reshape(2,1)
        ray = ProjectToWorld(projectionMatrix, p).reshape((-1))

        p1 = C
        p2 = ray

        rayList    = np.array([p1,p2],dtype=np.float32)
        intersectionFound  = tree.rayIntersection(rayList)

        sl = [pt.s for pt in intersectionFound]
        s_pos = [s for s in sl if s>0]

        if len(s_pos) > 0:

            smin = min(s_pos)
            i = sl.index(smin)

            pi = intersectionFound[i].p
            points_list.append(pi)

    return np.array(points_list)

def image_based_measurement(projectionMatrix, C, I, mesh_file, scale=1):
    # Read in stl file using vtk
    
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_file)
    reader.MergingOn()
    reader.Update()
    stl = reader.GetOutput()
    print("Number of points    = %d" % stl.GetNumberOfPoints())
    print("Number of triangles = %d" % stl.GetNumberOfCells())

    # Extract polygon info from stl

    # 1. Get array of point coordinates
    numPoints   = stl.GetNumberOfPoints()
    pointCoords = np.zeros((numPoints,3),dtype=float)
    for i in range(numPoints):
        pointCoords[i,:] = stl.GetPoint(i)
        
    # 2. Get polygon connectivity
    numPolys     = stl.GetNumberOfCells()
    connectivity = np.zeros((numPolys,3),dtype=np.int32)
    for i in range(numPolys):
        atri = stl.GetCell(i)
        ids = atri.GetPointIds()
        for j in range(3):
            connectivity[i,j] = ids.GetId(j)    

    # Create octree structure containing stl poly mesh
    tree = ot.PyOctree(pointCoords,connectivity)

    # create mesh actor
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    meshActor = vtk.vtkActor()
    meshActor.SetMapper(mapper)


    w2 = int(I.shape[1])
    h2 = int(I.shape[0])
    dim = (w2, h2)
    img = cv2.resize(I, dim)

    points_list=[]
    img_coords=[]

    def interactive_win(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            p = np.array([[u,v]]).T / scale
            ray = ProjectToWorld(projectionMatrix, p).reshape((-1))

            p1 = C
            p2 = ray

            rayList    = np.array([p1,p2],dtype=np.float32)
            intersectionFound  = tree.rayIntersection(rayList)

            sl = [pt.s for pt in intersectionFound]
            s_pos = [s for s in sl if s>0]

            if len(s_pos) > 0:

                smin = min(s_pos)
                i = sl.index(smin)

                pi = intersectionFound[i].p
                points_list.append(pi)
                img_coords.append([u,v])

                cv2.circle(img, (u, v), int(5*scale), (255, 0, 0), -1)

                if len(points_list) > 1:                    
                    u1=img_coords[-2][0]
                    v1=img_coords[-2][1]
                    p1=points_list[-2]

                    u2=img_coords[-1][0]
                    v2=img_coords[-1][1]
                    p2=points_list[-1]                     
                    cv2.line(img, (u1, v1), (u2, v2), (0, 255, 0), int(3*scale))     

                    cv2.putText(img,'%.3fm' % float(norm(p2-p1)), 
                        (int(u1/2+u2/2),int(v1/2+v2/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,(255,255,255),2)                                

                cv2.circle(img, (u, v), int(5*scale), (255, 0, 0), -1)


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', interactive_win)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 'Esc' Key
            return
        if k == 13:  # 'Enter' Key
            cv2.destroyAllWindows()
            break

    return np.array(img_coords), np.array(points_list)


def visualize_mesh(mesh_file,projectionMatrixList,CameraList,RotList,imgs=[]):    
    # Read in stl file using vtk
    
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_file)
    reader.MergingOn()
    reader.Update()
    stl = reader.GetOutput()
    print("Number of points    = %d" % stl.GetNumberOfPoints())
    print("Number of triangles = %d" % stl.GetNumberOfCells())

    # Extract polygon info from stl

    # 1. Get array of point coordinates
    numPoints   = stl.GetNumberOfPoints()
    pointCoords = np.zeros((numPoints,3),dtype=float)
    for i in range(numPoints):
        pointCoords[i,:] = stl.GetPoint(i)
        
    # 2. Get polygon connectivity
    numPolys     = stl.GetNumberOfCells()
    connectivity = np.zeros((numPolys,3),dtype=np.int32)
    for i in range(numPolys):
        atri = stl.GetCell(i)
        ids = atri.GetPointIds()
        for j in range(3):
            connectivity[i,j] = ids.GetId(j)    

    # Create octree structure containing stl poly mesh
    tree = ot.PyOctree(pointCoords,connectivity)

    # create mesh actor
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    meshActor = vtk.vtkActor()
    meshActor.SetMapper(mapper)

    camera_actors = []
    
    for i,(P,C,R) in enumerate(zip(projectionMatrixList,CameraList,RotList)):

        I=imgs[i]
        img_coords,_ = image_based_measurement(mesh_file,P, I, C)

        px,py,pz = C
        rx,ry,rz = R.dot([[0],[0],[-1]])

        # add cone
        cone = vtk.vtkConeSource()
        cone.SetResolution(100)

        cone.SetCenter(px,py,pz)
        cone.SetDirection(rx,ry,rz)
        cone.SetHeight(0.2)
        cone.SetRadius(0.1)

        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInputConnection(cone.GetOutputPort())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)
        coneActor.GetProperty().SetColor(1.0, 0.0, 0.0) #(R,G,B)
        camera_actors.append(coneActor)

        for p in img_coords:
            px = p[0]
            py = p[1]

            p = np.array([[px,py]]).T
            ray = ProjectToWorld(P, p).reshape((-1))

            p1 = C
            p2 = 1*ray

            rayList    = np.array([p1,p2],dtype=np.float32)
            intersectionFound  = tree.rayIntersection(rayList)            

            # create line source
            lineSource = vtk.vtkLineSource()
            lineSource.SetPoint1(p1)
            lineSource.SetPoint2(p2)
            lineSource.Update()

            # mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(lineSource.GetOutputPort())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color actor
            actor.GetProperty().SetColor(1,0,1)

            camera_actors.append(actor)

            for pt in intersectionFound:
                if pt.s < 0:
                    continue
                pi = pt.p
                px,py,pz = pi
                # create sphere source
                sphereSource = vtk.vtkSphereSource()
                sphereSource.SetCenter(px,py,pz)
                sphereSource.SetRadius(0.05)
                sphereSource.SetPhiResolution(50)
                sphereSource.SetThetaResolution(50)

                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphereSource.GetOutputPort())

                # actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # color actor
                actor.GetProperty().SetColor(1,0,1)

                camera_actors.append(actor)

                # only show 1st intersection
                break

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    
    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actors to the renderer
    ren.AddActor(meshActor)
    for c in camera_actors:
        ren.AddActor(c)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


if __name__ == '__main__':
    pass
