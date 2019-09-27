import numpy as np

def estimate_marchingsquare(data , threshold ):
    width = data.shape[0]
    height= data.shape[1]
    f,u,chi=0 ,0,0
    for i in range(width-1 ):
        for j in range(height-1):
            pattern=0
            if (data[i,j]     > threshold) : pattern += 1;
            if (data[i+1,j]   > threshold) : pattern += 2;
            if (data[i+1,j+1] > threshold) : pattern += 4;
            if (data[i,j+1 ]  > threshold) : pattern += 8;

            if pattern ==0 :
                break

            elif pattern==1:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j])
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi + 0.25;
                break;
            elif pattern==2:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1, (j+1)]);
                f = f + 0.5 * (1 - a1) * (a2);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi + 0.25;
                break;
            elif pattern==3:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,(j+1)]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + a2 + 0.5 * (a4 - a2);
                u = u + np.sqrt(1 + (a4 - a2) * (a4 - a2));
                break;
            elif pattern==4:
                a2 = (data[ i+1,j] - threshold) / (data[i+1,j ] - data[ i+1,j+1]);
                a3 = (data[ i,j+1 ] -  threshold) / (data[i,j+1] - data[ i+1,j+1]);
                f = f + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.25;
                break;
            elif pattern==5:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.5;
                break;
            elif pattern==6:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + (1 - a3) + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                break;
            elif pattern==7:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi - 0.25;
                break;

            elif pattern==8:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.25;
                break;

            elif pattern==9:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + a1 + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                break;

            elif pattern==10:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt(a1 * a1 + a4 * a4) + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.5;
                break;

            elif pattern==11:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + 1 - 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi - 0.25;
                break;

            elif pattern==12:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + (1 - a2) + 0.5 * (a2 - a4);
                u = u + np.sqrt(1 + (a2 - a4) * (a2 - a4));
                break;

            elif pattern==13:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                f = f + 1 - .5 * (1 - a1) * a2;
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi - 0.25;
                break;

            elif pattern==14:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi - 0.25;
                break;

            elif pattern == 15:
                f +=1 ;
                break;


    return f,u, chi

def get_functionals(im , nevals= 32):
    vmin =im.min() ; vmax=im.max()

    rhos =  pl.linspace( vmin,vmax, nevals)
    f= pl.zeros_like(rhos)
    u= pl.zeros_like(rhos)
    chi= pl.zeros_like(rhos)

    for k, rho in np.ndenumerate( rhos) :
        f[k], u[k],chi[k]=  estimate_marchingsquare(im, rho )

    return rhos, f,u,chi


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection
