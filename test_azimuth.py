#Calculates azimuths between two points
def azimuth(Lon1,Lat1,Lon2,Lat2):
    import nvector as nv
    wgs84 = nv.FrameE(name='WGS84')
    pointA = wgs84.GeoPoint(latitude=Lat1, longitude=Lon1, z=0, degrees=True)
    pointB = wgs84.GeoPoint(latitude=Lat2, longitude=Lon2, z=0, degrees=True)
    p_AB_N = pointA.delta_to(pointB)
    azim = p_AB_N.azimuth_deg
    #print(azim)
    if azim < 0:
        azim += 360
    else:
        pass
    return azim

