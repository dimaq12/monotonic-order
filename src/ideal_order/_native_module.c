#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * The Python layer uses ctypes against this extension's shared-object file.
 * The actual ideal_order_* symbols are compiled in from ideal_order.c as a
 * second Extension source. This tiny module supplies the required PyInit
 * entry point while keeping the public C API usable by non-Python callers.
 */

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native IdealOrder core.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&native_module);
}
