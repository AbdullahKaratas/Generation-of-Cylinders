import time
from cylinder_fitting import fitting_rmsd
from cylinder_fitting import geometry
from cylinder_fitting import show_G_distribution
from cylinder_fitting import show_fit
from cylinder_fitting import fit
import pandas as pd

import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.seterr(all='raise')


def extract_features(dataframe):
    extract_features = dataframe[dataframe.columns[7:]].copy()
    extract_features.columns = np.arange(0, len(extract_features.columns))

    return extract_features


def test_fit(cylinder_data, plotVisable):
    print("test fit.")

    C = np.array([0, 0, 0])
    r = 10

    data = cylinder_data

    start = time.time()
    w_fit, C_fit, r_fit, fit_err = fit(data)
    end = time.time()
    print('Fitting time: ', end - start)

    if plotVisable == True:
        show_fit(w_fit, C_fit, r_fit, data)

    return w_fit, r_fit, C_fit, fitting_rmsd(w_fit, C_fit, r_fit, data)


def rotationsmatrix(r, s, c):
    r = r / np.linalg.norm(r)
    Rot = np.array([[c+(1.-c)*r[0]*r[0], r[0]*r[1]*(1.-c)-r[2]*s, r[0]*r[2]*(1.-c)+r[1]*s],
                    [r[1]*r[0]*(1.-c)+r[2]*s, c + (1.-c)*r[1] *
                     r[1], r[1]*r[2]*(1.-c)-r[0]*s],
                    [r[2]*r[0]*(1.-c)-r[1]*s, r[2]*r[1]*(1.-c)+r[0]*s, c+r[2]*r[2]*(1.-c)]])

    return Rot


def kos_trafo(nx, ny, nz):
    e_1 = np.array([1., 0., 0.])
    e_2 = np.array([0., 1., 0.])
    e_3 = np.array([0., 0., 1.])

    normal = np.array([nx, ny, nz])
    normal = normal / np.linalg.norm(normal)
    r = np.cross(e_3, normal)
    zero = np.array([0., 0., 0.])

    if np.all(r == zero):
        pass

    else:

        s = np.linalg.norm(r)
        c = np.dot(normal, e_3)

        Rot = rotationsmatrix(r, s, c)

        e_1 = np.matmul(Rot, e_1)
        e_2 = np.matmul(Rot, e_2)
        e_3 = np.matmul(Rot, e_3)

    return e_1, e_2, e_3


def cylinder_calculator(normal, R_set, xc_set, yc_set, zc_set, n, n_p, deltaphi, L, alpha):
    cylinder_data = []

    theta_start_set = np.random.uniform(0., 2*np.pi, n)

    for k in range(n):
        e_1, e_2, e_3 = kos_trafo(normal[0], normal[1], normal[2])

        theta_start = - deltaphi/2
        theta_end = + deltaphi/2

        theta_set = np.random.uniform(theta_start, theta_end, n_p)
        z_area = np.random.uniform(-L / 2., L / 2., n_p)
        R = np.random.normal(R_set[k], alpha*R_set[k], n_p)
        cylinder_data_row = [R_set[k], normal[0], normal[1],
                             normal[2], xc_set[k], yc_set[k], zc_set[k]]

        measurements = []

        for i in range(n_p):

            term_x = R[i]*np.cos(theta_set[i]) * e_1[0] + R[i] * \
                np.sin(theta_set[i]) * e_2[0] + z_area[i] * e_3[0] + xc_set[k]
            term_y = R[i]*np.cos(theta_set[i]) * e_1[1] + R[i] * \
                np.sin(theta_set[i]) * e_2[1] + z_area[i] * e_3[1] + yc_set[k]
            term_z = R[i]*np.cos(theta_set[i]) * e_1[2] + R[i] * \
                np.sin(theta_set[i]) * e_2[2] + z_area[i] * e_3[2] + zc_set[k]

            measurements.append((term_x, term_y, term_z))

        measurements = sorted(measurements, key=lambda w: w[2])
        measurements = [r for s in measurements for r in s]

        cylinder_data_row.extend(measurements)
        cylinder_data.append(cylinder_data_row)

    column_list = ['R', 'nx', 'ny', 'nz', 'xc', 'yc', 'zc']
    column_list.extend(list(range(3 * n_p)))

    cylinder_data = pd.DataFrame(cylinder_data, columns=column_list)

    return cylinder_data, R_set


def starting_generation_and_fit(nReplay, nheight, ndata, lengthStep, Ropt, plotVisable):
    print('start')

    normalList = np.empty((nReplay, 3))
    directionList = np.empty((nReplay, 3))

    cReal = np.empty((nReplay, 3))
    cFit = np.empty((nReplay, 3))

    RadiusList = np.empty((nReplay, 1))
    RealRadiusList = np.empty((nReplay, 1))

    RMSDList = np.empty((nReplay, 1))

    for i in range(nReplay):
        normal = np.array([0, 0, 1]) + np.random.uniform(-1., 1., 3) * 1/1000
        normal = normal / np.linalg.norm(normal)
        normalList[i, :] = normal

        deltaStep = 0
        DataEnd = np.zeros((ndata*ndata*lengthStep*nheight, 3))

        R_min = Ropt*(1.-1/1000000)
        R_max = Ropt*(1.+1/1000000)
        xc_min = -1./1000
        xc_max = 1./1000
        yc_min = -1./1000
        yc_max = 1./1000

        R_set = np.random.uniform(R_min, R_max, 1)
        xc_set = np.random.uniform(xc_min, xc_max, 1)
        yc_set = np.random.uniform(yc_min, yc_max, 1)

        RealRadiusList[i, :] = R_set

        for j in range(nheight):
            zc_min = 2*j - 1./1000
            zc_max = 2*j+1./1000
            zc_set = np.random.uniform(zc_min, zc_max, 1)
            cReal[i, :] = [xc_set, yc_set, zc_set]
            print(cReal)
            for k in range(0, lengthStep):
                deltaStep = k*(360/lengthStep) * np.pi/180
                nameCSV = './cylinder_data_' + \
                    '{}'.format(k+lengthStep*j) + '.csv'
                saveDataIn = nameCSV
                data, Rset = cylinder_calculator(
                    normal, R_set, xc_set, yc_set, zc_set, n=1, n_p=ndata*ndata, deltaphi=0.8/Ropt, L=0.8, alpha=0.001)
                data.to_csv(saveDataIn, index=False)
                cylinder_data = pd.read_csv(
                    './cylinder_data_{}.csv'.format(k+lengthStep*j), sep=",")
                features_dataframe = extract_features(cylinder_data)
                features = np.array(features_dataframe,
                                    dtype=np.float32).reshape((ndata*ndata, 3))
                x = features[:, 0]
                y = features[:, 1]
                z = features[:, 2]
                X = x * np.cos(deltaStep) - y * np.sin(deltaStep)
                Y = x * np.sin(deltaStep) + y * np.cos(deltaStep)
                Z = z

                Data = np.vstack((X, Y, Z)).T
                start = j*lengthStep*ndata*ndata + ndata*ndata*k
                stop = j*lengthStep*ndata*ndata + ndata*ndata*(k+1)
                DataEnd[start:stop, :] = Data

        DataEnd_data = pd.DataFrame(DataEnd)
        DataEnd_data.to_csv(
            './Data_{}.csv'.format(i), index=False)

        normalList_data = pd.DataFrame(normalList)
        normalList_data.to_csv(
            './normalList_{}.csv'.format(i), index=False)

        RealRadiusList_data = pd.DataFrame(RealRadiusList)
        RealRadiusList_data.to_csv(
            './radiusList_{}.csv'.format(i), index=False)

        cReal_data = pd.DataFrame(cReal)
        cReal_data.to_csv(
            './center_{}.csv'.format(i), index=False)

        print('fit')
        w_fit, r_fit, c_fit, RMSD = test_fit(DataEnd, plotVisable)

        directionList[i, :] = w_fit
        RadiusList[i, :] = r_fit
        cFit[i, :] = c_fit

        directionList_data = pd.DataFrame(directionList)
        directionList_data.to_csv(
            './fit_direction_{}.csv'.format(i), index=False)

        RadiusList_data = pd.DataFrame(RadiusList)
        RadiusList_data.to_csv(
            './fit_RadiusList_{}.csv'.format(i), index=False)

        cFit_data = pd.DataFrame(cFit)
        cFit_data.to_csv(
            './fit_center_{}.csv'.format(i), index=False)

        print('run number = ', i)


if __name__ == "__main__":
    nReplay = 1
    nheight = 1
    ndata = 50
    lengthStep = 4
    Ropt = 12
    plotVisable = True
    starting_generation_and_fit(
        nReplay, nheight, ndata, lengthStep, Ropt, plotVisable)
