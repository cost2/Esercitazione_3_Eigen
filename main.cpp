#include <iostream>
#include <Eigen/Eigen>
#include <iomanip>

using namespace Eigen;
using namespace std;


//decomposizione PALU
Vector2d PALU(const Matrix2d &A, const Vector2d &b)
 {
    PartialPivLU<Matrix2d> lu(A);
    return lu.solve(b);
 }	
	
//decomposizione QR
Vector2d QR(const Matrix2d &A, const Vector2d &b)
{
    HouseholderQR<Matrix2d> qr(A);
    return qr.solve(b);
}

//calcolo errore relativo
double ErroreRelativo(const Vector2d &x_cal, const Vector2d &x_sol) 
{
    return (x_cal-x_sol).norm() / x_sol.norm();
}

int main()
{
	Matrix2d A1, A2, A3;
	
    A1 <<  5.547001962252291e-01, -3.770900990025203e-02,
   		   8.320502943378437e-01, -9.992887623566787e-01;
	
	A2 <<  5.547001962252291e-01, -5.540607316466765e-01,
		   8.320502943378437e-01, -8.324762492991313e-01;
	
	A3 <<  5.547001962252291e-01, -5.547001955851905e-01,
	       8.320502943378437e-01, -8.320502947645361e-01;
	
    Vector2d b1, b2, b3;
	
    b1 <<  -5.169911863249772e-01, 1.672384680188350e-01;
	
	b2 <<  -6.394645785530173e-04, 4.259549612877223e-04;
	
	b3 <<  -6.400391328043042e-10, 4.266924591433963e-10;
	
    Vector2d x_sol(-1.0, -1.0);
	
	for (int i=1; i<=3; i++) 
	{
        Matrix2d A = (i==1) ? A1 : (i==2) ? A2 : A3;
        Vector2d b = (i==1) ? b1 : (i==2) ? b2 : b3;

        Vector2d x_palu = PALU(A, b);
        Vector2d x_qr = QR(A, b);

        double err_palu = ErroreRelativo(x_palu, x_sol);
        double err_qr = ErroreRelativo(x_qr, x_sol);

        cout << "Sistema " << i << ":\n";
        cout << scientific << setprecision(1) << "Soluzione PALU: " << x_palu.transpose() << scientific << setprecision(5) << ", Errore Relativo: " << err_palu << "\n";
        cout << scientific << setprecision(1) << "Soluzione QR: " << x_qr.transpose() << scientific << setprecision(5) << ", Errore Relativo: " << err_qr << "\n\n";
    }
	
    return 0;
}
