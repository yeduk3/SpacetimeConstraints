//
//  main.cpp
//  SpacetimeConstraints
//
//  Created by 이용규 on 11/2/25.
//

#include <YGLWindow.hpp>
YGLWindow *window;
#include <program.hpp>
Program shader;
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <limits>

using namespace std;
using namespace Eigen;

//******************************
//      Animation Setting
//******************************

// Time binning
const float T = 5.f;
const int N = 5;
const float h = T/N;
// mass(kg)
const int m = 1;
// Start position a and End position b
Vector3f a(-0.5f, -0.5f, 0), b(0.5f, 0.5f, 0);

// gravity acceleration
const float g = -9.8f;

//***********************************
//    Animating Point Structure
//***********************************

struct ObjectFraming {
    float animatingTime; // 1 cycle time
    int numFrame; // total frame = position.size() / 3
    vector<float> position; // vector of 3d vectors
    vector<float> velocity; // vector of 3d vectors
    GLuint vao = -1, posvbo[2], velvbo[2];
    int animatingFrame = -1;
    static ObjectFraming create(const float& animatingTime_, const vector<float>& position_, const vector<float>& velocity_) { return ObjectFraming(animatingTime_, position_, velocity_); }
    
    ObjectFraming() {}
    ObjectFraming(const float& animatingTime_, const vector<float>& position_, const vector<float>& velocity_) : animatingTime(animatingTime_), position(position_), velocity(velocity_) {
        numFrame = int(position_.size()) / 3;
        
        cout << "--- Key Frame Object Animation ---" << endl;
        cout << "animating Time: " << animatingTime << endl;
        cout << "numFrame:       " << numFrame << endl;
        
        cout << "----------------------------------" << endl;
    }
    void render(const float& t, bool loop = true) {
        int currFrame, nextFrame;
        float tt = t/animatingTime;
        float timeRatio = (tt - floor(tt))*numFrame;
        if(loop) {
            currFrame = int(timeRatio) % numFrame;
            nextFrame = (currFrame + 1) % numFrame;
        } else {
            currFrame = t > animatingTime ? numFrame-1 : int(timeRatio);
            nextFrame = currFrame == numFrame-1 ? currFrame : currFrame+1;
        }
        
        shader.use();
        shader.setUniform("interTime", min(1.f, max(0.f, timeRatio - currFrame)));
        
        if(vao == -1) {
            vector<float> currpos(position.begin(),   position.begin()+3);
            vector<float> nextpos(position.begin()+3, position.begin()+6);
            vector<float> currvel(velocity.begin(),   velocity.begin()+3);
            vector<float> nextvel(velocity.begin()+3, velocity.begin()+6);
            glGenVertexArrays(1, &vao);
            glGenBuffers(2, posvbo);
            glBindBuffer(GL_ARRAY_BUFFER, posvbo[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3, currpos.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, posvbo[1]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3, nextpos.data(), GL_DYNAMIC_DRAW);
            glGenBuffers(2, velvbo);
            glBindBuffer(GL_ARRAY_BUFFER, velvbo[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3, currpos.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, velvbo[1]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3, nextpos.data(), GL_DYNAMIC_DRAW);
        }
        
        if(animatingFrame != currFrame) {
//            cout << "Frame: " << curFrame << ", " << nextFrame << endl;
            animatingFrame = currFrame;
            vector<float> currpos(position.begin()+currFrame*3, position.begin()+currFrame*3+3);
            vector<float> nextpos(position.begin()+nextFrame*3, position.begin()+nextFrame*3+3);
            vector<float> currvel(velocity.begin()+currFrame*3, velocity.begin()+currFrame*3+3);
            vector<float> nextvel(velocity.begin()+nextFrame*3, velocity.begin()+nextFrame*3+3);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, posvbo[0]); // fill 0
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3, currpos.data());
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, posvbo[1]); // fill 1
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3, nextpos.data());
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, velvbo[0]); // fill 2
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3, currvel.data());
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, velvbo[1]); // fill 3
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3, nextvel.data());
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
        }
        glBindVertexArray(vao);
        
        glPointSize(30.f);
        
        glDrawArrays(GL_POINTS, 0, 1);
    }
};
ObjectFraming obj;

//******************************
//   Linear System Variables
//******************************

LeastSquaresConjugateGradient<SparseMatrix<float>> lscg;
ConjugateGradient<SparseMatrix<float>, Lower|Upper> cg;

// Denotes that the expression `i` means the i-th discretized time step.

// x, y and z values of the position x_i's and the force f_i's
// ex) [x_1, x_2, ..., x_n, f_1, f_2, ..., f_n]^T, and each x_i, f_i is 3d vector.
VectorXf S;
// C_i consists of the physical constraints p_i, boundary constraint c_a and c_b, and additional boundary frame stop conditions v_1, v_n
// p_i differs by cases. defined in (1, n)
// c_a = x_1 - a = 0 for the start point a
// c_b = x_n - b = 0 for the end point b
// v_1 = x_2 - x_1 = 0
// v_n = x_n-x_{n-1} = 0
// [v_1, c_a, p_2, ..., p_{n-1}, c_b, v_n]^T
VectorXf C;
// Jacobian matrix of the constraints C_i
SparseMatrix<float> dCdS;
// R is the squared sum of all f_i's (multiplied by h) and should be minimized.
// Jacobian and Hessian matrix of the objective function R
VectorXf dRdS;
SparseMatrix<float> dRdS2;

const int n = N*3; // number of elements of x_1 to x_n

using TriList = vector<Triplet<float>>;

void put33DiagToTriList(TriList& triplets, int row, int col, float v1, float v2, float v3) {
    triplets.emplace_back(row, col, v1);
    triplets.emplace_back(row+1, col+1, v2);
    triplets.emplace_back(row+2, col+2, v3);
}

void put33DiagToTriList(TriList& triplets, int row, int col, float v) {
    put33DiagToTriList(triplets, row, col, v, v, v);
}

void construct_C() {
    C = VectorXf::Zero(n+9);
    // v_1: x_2 - x_1 = 0
    C.segment<3>(0) = S.segment<3>(3) - S.segment<3>(0);
    // c_a: x_1 - a = 0
    C.segment<3>(3) = S.segment<3>(0) - a;
    // p_i: m*(x_{i-1} - 2*x_i + x_{i+1})/h^2 - f_i - mg
    VectorXf gravity = VectorXf::Zero(n-6);
    for(int i = 1; i < gravity.rows(); i += 3) gravity.coeffRef(i) = m*g;
    C.segment<n-6>(6) = m*( S.segment<n-6>(0) - 2*S.segment<n-6>(3) + S.segment<n-6>(6) )/(h*h) - S.segment<n-6>(n+3) - gravity;
    // c_b: x_n - b = 0
    C.segment<3>(n) = S.segment<3>(n-3) - b;
    // v_n: x_n - x_{n-1} = 0
    C.segment<3>(n+3) = S.segment<3>(n-3) - S.segment<3>(n-6);
    // test additional
    C.segment<3>(n+6) = S.segment<3>(6) - Vector3f(0, 0.5, 0);

    cout << "C(" <<  C.rows() << ", " << C.cols() << ")" << endl;
}

TriList construct_dCdS() {
    TriList triplets;
    if(C.rows() == 0) construct_C();
    dCdS = SparseMatrix<float>(C.rows(), S.rows());
    
    // d(v_1)/d(x_1) = -1, d(v_1)/d(x_2) = 1
    put33DiagToTriList(triplets, 0, 0, -1); put33DiagToTriList(triplets, 0, 3, 1);
    // d(c_a)/d(x_1)
    put33DiagToTriList(triplets, 3, 0, 1);
    // d(p_i)/d(x_j)
    float lr = -m/(h*h), me = 2*m/(h*h);
    for(int i = 6; i < n; i += 3) {
        put33DiagToTriList(triplets, i, i-6, lr);
        put33DiagToTriList(triplets, i, i-3, me);
        put33DiagToTriList(triplets, i, i  , lr);
    }
    // d(c_b)/d(x_n)
    put33DiagToTriList(triplets, n, n-3, 1);
    // d(v_n)/d(x_{n-1}) = -1, d(v_n)/d(x_n) = 1
    put33DiagToTriList(triplets, n+3, n-6, -1); put33DiagToTriList(triplets, n+3, n-3, 1);
    // d(p_i)/d(f_j)
    for(int i = 6; i < n; i += 3) {
        put33DiagToTriList(triplets, i, i+n-3, -1);
    }
    put33DiagToTriList(triplets, n+6, 6, 1);
    
    dCdS.setFromTriplets(triplets.begin(), triplets.end());
    
    cout << "dCdS(" <<  dCdS.rows() << ", " << dCdS.cols() << ")" << endl;
    
    return triplets;
}

void construct_dRdS() {
    dRdS = VectorXf::Zero(2*n);
    
    // dRdx = 0?
    // dRdf = 2hf_i
    dRdS.segment<n>(n) = S.segment<n>(n)*2*h;
    
    cout << "dRdS(" <<  dRdS.rows() << ", " << dRdS.cols() << ")" << endl;
}

TriList construct_dRdS2() {
    dRdS2 = SparseMatrix<float>(2*n, 2*n);
    TriList triplets;
    
    // d(R)/(d(x)d(x)) = 0?
    // d(R)/(d(x)d(f)) = 0?
    // d(R)/(d(f)d(f)) = 2h
//    float h2 = 2*h;
    for(int i = n; i < 2*n; i += 3) {
        put33DiagToTriList(triplets, i, i, 2*h);
    }
    
    dRdS2.setFromTriplets(triplets.begin(), triplets.end());
    
    cout << "dRdS2(" <<  dRdS2.rows() << ", " << dRdS2.cols() << ")" << endl;
    
    return triplets;
}

SparseMatrix<float> KKT(0, 0);
VectorXf KKT_b(0);
VectorXf l;
// SQP solves the linear system of [H J^T \\ J 0][ds \\ dl] = [dRdS+J^Tl \\ C]
void construct_KKTLinearized() {
    construct_C();
    construct_dRdS();
    auto H_tri = construct_dRdS2();
    auto J_tri = construct_dCdS();
    KKT = SparseMatrix<float>(dRdS2.rows() + dCdS.rows(), dRdS2.cols() + dCdS.rows());
    
    TriList triplets;
    triplets.reserve(H_tri.size() + 2*J_tri.size());
    
    // H
    triplets.insert(triplets.end(), H_tri.begin(), H_tri.end());
    
    // J^T
    for(const auto& j : J_tri) triplets.emplace_back(j.col(), j.row()+dRdS2.cols(), j.value());
    
    // J
    for(const auto& j : J_tri) triplets.emplace_back(j.row()+dRdS2.rows(), j.col(), j.value());
    
    KKT.setFromTriplets(triplets.begin(), triplets.end());
    
    cout << "KKT(" <<  KKT.rows() << ", " << KKT.cols() << ")" << endl;
//    cout << KKT << endl;
    
    if(KKT_b.rows() == 0) l = VectorXf::Zero(dCdS.rows());/*l = KKT_b.segment(dRdS2.cols(), dCdS.rows());*/
    
    KKT_b = VectorXf(dRdS2.cols() + dCdS.rows());
    KKT_b.segment(0, dRdS2.cols()) = -dRdS - dCdS.transpose()*l;
    KKT_b.segment(dRdS2.cols(), dCdS.rows()) = -C;
}

// Sequential Quadratic Programming
void SQP(bool useKKT = false) {
    // Zero Starting S
    S = VectorXf::Zero(2*n);
    
    VectorXf prevS;
    float tolerance = 1e-4, prevCond = numeric_limits<float>::max();
    int cnt = 0, maxCnt = 100;
    while(++cnt < maxCnt) { // small enough C or iterated enough, stop iteration
        cout << cnt << "-th iteration:" << endl;
        
        
        VectorXf deltaS;
        if(!useKKT) {
            // Construct system //
            construct_C();
            {
                // end condition should be checked after constructing C
                cout << "  C norm: " << C.norm() << endl;
                float cond = C.norm();
                float Rnorm = 0.f;
                for(int i = 0; i < n; i += 3) Rnorm += S.segment<3>(i+n).norm();
                cout << "  R norm: " << Rnorm << endl;
                // end condition 1: C is small --> may be converged
                if(cond < tolerance) break;
                // end condition 2: further decrease in R requires violating the constraints
                if(cond > prevCond) { S = prevS; break; }
                prevCond = cond;
            }
            construct_dCdS();
            construct_dRdS();
            construct_dRdS2();
            
            // Solve //
            cg.compute(dRdS2.block(n, n, n, n));
            VectorXf S_hat = VectorXf::Zero(dRdS.rows());
            S_hat.segment<n>(n) += cg.solve(-dRdS.segment<n>(n));
            cout << "  #iterations:     " << cg.iterations() << endl;
            cout << "  estimated error: " << cg.error()      << endl;
            lscg.compute(dCdS);
            VectorXf C_modified = -C - dCdS*S_hat;
            VectorXf S_tilde = lscg.solve(C_modified);
            cout << "  #iterations:     " << lscg.iterations() << endl;
            cout << "  estimated error: " << lscg.error()      << endl;
            deltaS = S_hat+S_tilde;
        }
        else {
            // Construct system //
            construct_KKTLinearized();
            {
                // end condition should be checked after constructing C
                cout << "  C norm: " << C.norm() << endl;
                float cond = C.norm();
                float Rnorm = 0.f;
                for(int i = 0; i < n; i += 3) Rnorm += S.segment<3>(i+n).norm();
                cout << "  R norm: " << Rnorm << endl;
                // end condition 1: C is small --> may be converged
                if(cond < tolerance) break;
                // end condition 2: further decrease in R requires violating the constraints
                if(cond > prevCond) { S = prevS; break; }
                prevCond = cond;
            }
            
            // Solve //
            cout << "Try to solve" << endl;
            lscg.compute(KKT);
            auto d = lscg.solve(KKT_b);
            cout << "  iteration: " << lscg.iterations() << endl;
            cout << "  error: " << lscg.error() << endl;
            l += d.segment(S.rows(), dCdS.rows());
            deltaS = d.segment(0, S.rows());
        }
        prevS = S;
        S += deltaS;
        
        if(deltaS.norm() < tolerance) break; // No more update?
        
        // Check the results //
        for(size_t row = 0; row < S.rows(); row += 3) {
            cout<<"  ("<<S.coeff(row)<<","<<S.coeff(row+1)<<","<<S.coeff(row+2)<<")"<<endl;
        }
    }
    cout << "Stopped on cnt = " << cnt << ",  final result:" << endl;
    
}

// Sequential Quadratic Programming
//VectorXf SQP() {
//    cg.compute(dRdS2.block(n, n, n, n));
//    VectorXf S_hat = VectorXf::Zero(dRdS.rows());
//    S_hat.segment<n>(n) += cg.solve(-dRdS.segment<n>(n));
//    cout << "  #iterations:     " << cg.iterations() << endl;
//    cout << "  estimated error: " << cg.error()      << endl;
//    lscg.compute(dCdS);
//    VectorXf C_modified = -C - dCdS*S_hat;
//    VectorXf S_tilde = lscg.solve(C_modified);
//    cout << "  #iterations:     " << lscg.iterations() << endl;
//    cout << "  estimated error: " << lscg.error()      << endl;
//    return S_hat+S_tilde;
//}

bool renderLoop = false;
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if(key == GLFW_KEY_0 && action == GLFW_PRESS) {
        glfwSetTime(0);
    }
    else if(key == GLFW_KEY_R && action == GLFW_PRESS) {
        renderLoop = !renderLoop;
    }
}

void init() {
    SQP();
    
    
    for(size_t row = 0; row < S.rows(); row += 3) {
        cout<<"  ("<<S.coeff(row)<<","<<S.coeff(row+1)<<","<<S.coeff(row+2)<<")"<<endl;
    }
    
    shader.loadShader("vs.vert", "fs.frag");
    
    vector<float> pos(n), vel(n);
    float v = 0.f;
    for(int i = 0; i < n; i++) pos[i] = (S.coeff(i));
    VectorXf gravity = VectorXf::Zero(n);
    for(int i = 4; i < n-3; i+=3) gravity.coeffRef(i) = g;
    VectorXf acc = S.segment<n>(n)/m + gravity;
    cout << "v: " << endl;
    for(int i = 0; i < n; i++) {
        if(i % 3 == 0) cout << endl;
        vel[i] = v;
        v += acc.coeff(i)*h;
        cout << v << ", ";
    }
    cout << endl;
    obj = ObjectFraming::create(T, pos, vel);
    
    glfwSetKeyCallback(window->getGLFWWindow(), keyCallback);
}
float t = -1;
void render() {
    if(t < 0) glfwSetTime(0);
    glViewport(0, 0, window->width(), window->height());
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    t = glfwGetTime();
    obj.render(t, renderLoop);
}

int main(int argc, const char * argv[]) {
    
    window = new YGLWindow(640, 480, "Spacetime Constraints");
    
    window->mainLoop(init, render);
    
    return EXIT_SUCCESS;
}
