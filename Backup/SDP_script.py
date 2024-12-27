import utils as fc
import numpy as np

# Creating n qubit-states, with dimension d
n = 5
d = 4

# Creating the separable qubit-state
rho_sep = (np.eye(4))/4

# Creating the polytope of m projective dichotomic measurements
# if the first attribute of the function measurements is 1: m = 3, 
# if it is 2: m = 9, if it is 3: m = 13.
# eta is the inradius of the polytope
# if the second attribute is TRUE it will generate the png plot of the polytope
medicoes, eta = fc.measurements(1,PLOT=True)
m = int(medicoes.shape[0]/2)
print(eta)

# Creating the deterministic strategies
detp = fc.strategies_LHS(m,2)

# Creating the file to write the results
f1 = open("results_"+str(n)+".txt", "w")

# Creating the file to save the states
f2 = open("states_"+str(n)+".txt", "w")

# Creating the file to save fast info
f3 = open("info_"+str(n)+".txt","w")

for i in range(n):

    # Using the functions created (Bures or Hilbert-Schmidt)

    rho = fc.rho_mixed(d)

    #rho = fc.rho_mixed_HS(d) 

    f2.write(str(rho)+"\n")
    
    w,v,ppt = fc.Ent_cert(rho)

    input_message = [str('-'*84),"\nTarget state rho:\n",str(rho),"\nIs the state separable?\n"]

    f1.writelines(input_message)

    if ppt == 0:
        s = "Yes: separable state.\n"
    else:
        s = "No: entangled state.\n"

    f1.write(s)
    f3.write(str(i)+" "+str(ppt)+"\n")

    P,solution,q = fc.SDP_LHS(m,2,rho,rho_sep,eta,detp,medicoes)
    print(P)
    print(solution)

    rho_q = rho*q+(1-q)*rho_sep
    output_message = ("Optimal value from SDP\n",str(q),"\nResulting state\n",str(rho_q),"\nIs the state separable?\n")
    f1.writelines(output_message)
    w,v,ppt = fc.Ent_cert(rho_q)
    if ppt == 0:
        s = "Yes: separable state.\n"
    else:
        s = "No: entangled state.\n"
    f1.write(s)
    f3.write(" "+str(q)+" "+str(ppt)+"\n")

# Closing the files
f1.close()
f2.close()
f3.close()