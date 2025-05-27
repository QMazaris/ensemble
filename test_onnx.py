try:
    import onnx
    print("ONNX version:", onnx.__version__)
    print("ONNX path:", onnx.__file__)
    
    # Try importing related packages
    import skl2onnx
    print("skl2onnx version:", skl2onnx.__version__)
    
    from skl2onnx import convert_sklearn
    print("convert_sklearn import from skl2onnx is available")
    
except ImportError as e:
    print("Import Error:", str(e))
except Exception as e:
    print("Other Error:", str(e)) 