//
// Created by Robin Huang on 11/26/16.
//

#ifndef APPC_PYUTILS_H
#define APPC_PYUTILS_H
#include <Python.h>
#include <string>

void pydebug(PyObject * obj, const char * name, const char * debugstring, PyObject * globals = NULL, PyObject * locals = NULL);
void pyrun(const char * debugstring, PyObject * globals = NULL, PyObject * locals = NULL);

//class PyThreadBeginEnd
//{
//public:
//    PyThreadBeginEnd() {
//        py_threadstate = PyGILState_Ensure();
//        _save = PyEval_SaveThread();
//    }
//    ~PyThreadBeginEnd() {
//        PyEval_RestoreThread(_save);
//        PyGILState_Release(py_threadstate);
//    }
//private:
//    PyGILState_STATE py_threadstate;
//    PyThreadState *_save;
//};

class PyGILEnsure
{
public:
    PyGILEnsure() {
        py_threadstate = PyGILState_Ensure();
    }
    virtual ~PyGILEnsure()
    {
        PyGILState_Release(py_threadstate);
    }
protected:
    PyGILState_STATE py_threadstate;
};

class PyRun
{
public:
    PyRun(bool use_self_globals = true);
    virtual ~PyRun();
    virtual void addToLocal(const char * name, PyObject * object);
    virtual void addToGlobal(const char * name, PyObject * object);
    virtual PyObject * eval(const char * code, std::string * err_msg = nullptr);
    virtual void run(const char * code, std::string * err_msg = nullptr);
    virtual void interpreter(const char * code);
    virtual const std::string repr(PyObject * obj);
    virtual bool errorOccurred(std::string * err_msg = nullptr);
private:
    PyObject * m_globals = NULL;
    PyObject * m_locals = NULL;
    PyObject * _run_code(const char * code, int startcode = Py_file_input, std::string * err_msg = nullptr);
};
#endif //APPC_PYUTILS_H
