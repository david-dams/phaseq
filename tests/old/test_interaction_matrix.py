# transformations from nested loops to convolutions and masked dot products #
import jax
import jax.numpy as jnp
from phaseq import *

from test_interaction_sum import *

def inner_sum_ref(l1, l2, l3, l4, px, pax, pbx, qcx, qdx, g1, g2):
    delta = 1/(4*g1) + 1/(4*g2)
    ret = [0.0] * (l1+l2+l3+l4+1)
    for i1 in range(l1 + l2 + 1):
        for i2 in range(l3 + l4 + 1):
            for r1 in range(i1 // 2 + 1):
                for r2 in range(i2 // 2 + 1):
                    for u in range( (i1+i2) // 2 - r1 - r2 + 1):
                        I =  i1 + i2 - 2*(r1+r2) - u
                        term = 1
                        term *= binomial_prefactor_ref(i1, l1, l2, pax, pbx) * factorial(i1) / (4*g1)**i1
                        term *= (4*g1)**r1 / (factorial(r1) * factorial(i1 - 2*r1))
                        term *= (-1)**i2 * binomial_prefactor_ref(i2, l3, l4, qcx, qdx) * factorial(i2) / (4*g2)**i2
                        term *= (4*g2)**r2 / (factorial(r2) * factorial(i2 - 2*r2))
                        term *= factorial(i1+i2-2*(r1+r2)) * (-1)**u * np.pow(px, I - u)
                        term /= factorial(u)*factorial(I - u)*np.pow(delta, I)
                        ret[I] += float(term)
                        print(I, u, term)
    return jnp.array(ret)

def outer_sum_ref(gaussian1, gaussian2, gaussian3, gaussian4):
    a1, a2 = gaussian1[-1], gaussian2[-1]
    g1 = float(a1 + a2)
    p = (a1 * gaussian1[:3] + a2 * gaussian2[:3]) / (a1 + a2)
    
    a3, a4 = gaussian3[-1], gaussian4[-1]
    g2 = float(a3 + a4)
    q = (a3 * gaussian3[:3] + a4 * gaussian4[:3]) / (a3 + a4)

    px, py, pz = q - p
    pax, pay, paz = p - gaussian1[:3]
    pbx, pby, pbz = p - gaussian2[:3]
    qcx, qcy, qcz = q - gaussian3[:3]
    qdx, qdy, qdz = q - gaussian4[:3]
    
    gx = inner_sum_ref(int(gaussian1[3]), int(gaussian2[3]), int(gaussian3[3]), int(gaussian4[3]), px, pax, pbx, qcx, qdx, g1, g2)
    gy = inner_sum_ref(int(gaussian1[4]), int(gaussian2[4]), int(gaussian3[4]), int(gaussian4[4]), py, pay, pby, qcy, qdy, g1, g2)
    gz = inner_sum_ref(int(gaussian1[5]), int(gaussian2[5]), int(gaussian3[5]), int(gaussian4[5]), pz, paz, pbz, qcz, qdz, g1, g2)

    print(gx, gy, gz)

    lgx, lgy, lgz = len(gx), len(gy), len(gz)

    delta = 1/(4*g1) + 1/(4*g2)
    arg = np.linalg.norm(p-q)**2 / (4 * delta)
    
    integrator = PyQInt()
    gx1 = integrator.B_array(int(gaussian1[3]), int(gaussian2[3]), int(gaussian3[3]), int(gaussian4[3]), float(p[0]), float(gaussian1[0]), float(gaussian2[0]), float(q[0]), float(gaussian3[0]), float(gaussian4[0]), g1, g2, delta)
    gy1 = integrator.B_array(int(gaussian1[4]), int(gaussian2[4]), int(gaussian3[4]), int(gaussian4[4]), float(p[1]), float(gaussian1[1]), float(gaussian2[1]), float(q[1]), float(gaussian3[1]), float(gaussian4[1]), g1, g2, delta)
    gz1 = integrator.B_array(int(gaussian1[5]), int(gaussian2[5]), int(gaussian3[5]), int(gaussian4[5]), float(p[2]), float(gaussian1[2]), float(gaussian2[2]), float(q[2]), float(gaussian3[2]), float(gaussian4[2]), g1, g2, delta)
    print(gx1, gy1, gz1, len(gx1), len(gy1), len(gz1))
        
    rab2 = np.linalg.norm(gaussian1[:3]-gaussian2[:3])**2
    rcd2 = np.linalg.norm(gaussian3[:3]-gaussian4[:3])**2
    prefac = 2.0 * np.pow(np.pi, 2.5) / (g1 * g2 * np.sqrt(g1 + g2)) * np.exp(-a1*a2*rab2/g1) * np.exp(-a3*a4*rcd2/g2)
    
    conv = jnp.convolve(jnp.convolve(gx, gy), gz)
    
    # function array over unique range
    f_arr = boys_fun( jnp.arange(conv.size), arg )

    return prefac * conv @ f_arr

if __name__ == '__main__':
    
    a1, l1, m1, n1, x1, y1, z1 =  0.5, 4, 0, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 0, 0, 10., 0.1, 0.5
    a3, l3, m3, n3, x3, y3, z3 =  0.3, 4, 0, 2, 0.4, 10, 0.9
    a4, l4, m4, n4, x4, y4, z4 =  0.8, 2, 0, 0, 1., 5, 0.3

    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    gaussian3 = jnp.array( [x3, y3, z3, l3, m3, n3, a3] )
    gaussian4 = jnp.array( [x4, y4, z4, l4, m4, n4, a4] )

    val = interaction(gaussian1, gaussian2, gaussian3, gaussian4)
    ref = outer_sum_ref(gaussian1, gaussian2, gaussian3, gaussian4) 
    import pdb; pdb.set_trace()

    
    val = interaction(gaussian1, gaussian1, gaussian1, gaussian1) 
    
    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)
    gto3 = gto(1., [x3, y3, z3], a3, l3, m3, n3)
    gto4 = gto(1., [x4, y4, z4], a4, l4, m4, n4)
    ref = integrator.repulsion_gto(gto1, gto1, gto1, gto1)
