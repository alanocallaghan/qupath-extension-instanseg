# inputs["ndarray"] is automagically assigned in the global python environment
ins_output = ins.eval_medium_image(ndarray.ndarray(), return_image_tensor=False, pixel_size=pixel_size).numpy()

# appose's shared memory array from the python side
## todo type
nda = NDArray(dtype="uint16", shape=ins_output.shape)
# create a np.ndarray-like view on the ndarray buffer
nda_np = nda.ndarray()
# copy the instanseg output into the ndarray buffer
nda_np[:] = ins_output
# return the ndarray view on the shared memory
task.outputs["instanseg_output"] = nda

