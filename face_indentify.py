import face_recognition
from PIL import Image, ImageDraw

#   located the known image (Bill Gates, Elon Musk, Steve Jobs, Mark Zuckerberg, Jeff Bezos) from known folder
#   encoding the image that can be use for face matching
gates_image = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/known/Bill Gates.jpg')
gates_encoding = face_recognition.face_encodings(gates_image)[0]

musk_image = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/known/Elon Musk.jpg')
musk_encoding = face_recognition.face_encodings(musk_image)[0]

jobs_image = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/known/Steve Jobs.jpg')
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

zuckerberg_image = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/known/Mark Zuckerberg.jpeg')
zuckerberg_encoding = face_recognition.face_encodings(zuckerberg_image)[0]

bezos_image = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/known/Jeff Bezos.jpeg')
bezos_encoding = face_recognition.face_encodings(bezos_image)[0]

#  Create array of encodings
known_face_encodings = [
  gates_encoding,
  musk_encoding,
  jobs_encoding,
  zuckerberg_encoding,
  bezos_encoding
  
]

#   Create array of known names
known_face_names = [
  "Bill Gates",
  "Elon Musk",
  "Steve Jobs",
  "Mark Zuckerberg",
  "Jeff Bezos"
  
]

# Load group image to find faces in
group_faces = face_recognition.load_image_file('/home/deeplearning/Downloads/VBH_FINALCODE/img/groups/7great.png')

# Find faces in test image
group_locations = face_recognition.face_locations(group_faces)
group_encodings = face_recognition.face_encodings(group_faces, group_locations)

# Convert to PIL format
pil_image = Image.fromarray(group_faces)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(group_locations, group_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
  #who does not have face in the known encoding array will be count as Unknown Person
  name = "Unknown Person"

  # If match with the known encoding face named with known face name arrays
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('face_match.jpg')