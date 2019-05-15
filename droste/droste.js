var cubeRotation = 0.0;
var gridScale = 1.0;

main();

//
// Start here
//
function main() {
    const canvas = document.querySelector('#glcanvas');
    const gl = canvas.getContext('webgl', {antialias: true});
    gl.getExtension('OES_standard_derivatives');
    gl.getExtension('EXT_shader_texture_lod');

    // If we don't have a GL context, give up now

    if (!gl) {
	alert('Unable to initialize WebGL. Your browser or machine may not support it.');
	return;
    }

    // Vertex shader program

    const vsSource = `
    attribute vec4 aVertexPosition;
    attribute vec2 aTextureCoord;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;

    varying highp vec2 vTextureCoord;

    void main(void) {
      gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;
      vTextureCoord = aTextureCoord;
    }
  `;

    // Fragment shader program

    const fsSource = `
    #extension GL_EXT_shader_texture_lod : enable
    #extension GL_OES_standard_derivatives : enable
    varying highp vec2 vTextureCoord;

    uniform sampler2D uSampler;

    uniform highp float uScale;

    uniform highp vec2 alpha;

    highp float lookup(vec2 p) {
      p *= uScale;

      highp float n, nx, ny;

      highp float x, y;

      nx = floor(log2(abs(p.x)));
      ny = floor(log2(abs(p.y)));

      n = max(nx, ny);

      p.x /= exp2(n);
      p.y /= exp2(n);

      const highp float PI = 3.1415926535897932384626433832795;
      highp float theta = atan(p.y, p.x);
      highp float r  = length(p);

      highp float gx = fract(p.x * 4.0 + 0.025);
      highp float gy = fract(p.y * 4.0 + 0.025);
      highp float g  = min(gx, gy);

      highp float hx = fract(p.x + 0.01);
      highp float hy = fract(p.y + 0.01);
      highp float h  = min(hx, hy);

      highp float r_grid     = fract(r * 5.0 + 0.025);
      highp float theta_grid = fract(theta * 12.0 / (2.0*PI) + 0.01);

      if (h < 0.02) {
        return 1.0;
      } else if (g < 0.05) {
        return 1.0;
      } else {
        return 0.0;
      }

      /*
      else if (r_grid < 0.05) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      } else if (theta_grid < 0.02) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      } else {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
      }
      */
    }

    highp vec4 lookup_droste(vec2 p) {
      p *= uScale;

      highp vec2 q = p;

      highp float n;
      highp vec2 f, l;
      l.x = log2(abs(p.x));
      l.y = log2(abs(p.y));
      f.x = floor(l.x);
      f.y = floor(l.y);

      n = max(f.x, f.y);
      p.x /= 4.0*exp2(n);
      p.y /= 4.0*exp2(n);

      highp vec2 p2 = p/2.0;

      p.x = (p.x + 0.5);
      p.y = 1.0 - (p.y + 0.5);

      p2.x = (p2.x + 0.5);
      p2.y = 1.0 - (p2.y + 0.5);

      highp vec4 s;
      highp vec4 t;

      s = texture2DGradEXT(uSampler, p,
                           dFdx(q), dFdy(q));

      t = texture2DGradEXT(uSampler, p2,
                           dFdx(q), dFdy(q));

      /*
      highp vec4 s = textureGrad(uSampler, p);
      highp vec4 t = textureGrad(uSampler, p2);
      */

      highp vec4 c;
      c = vec4(s.a) * s + vec4(1.0 - s.a) * t;
      return c;
    }

    void main(void) {
      highp float count = 0.0;
      highp float dp = 0.0005;

      // highp vec2 p = vTextureCoord;

      /*
      highp vec2 p;
      p.x = atan(vTextureCoord.y, vTextureCoord.x);
      p.y = length(vTextureCoord);
      */

      highp float r = length(vTextureCoord);
      highp float theta = atan(vTextureCoord.y, vTextureCoord.x);
      highp float c = 1.0; //alpha.x;
      highp float d = 0.1103178000763258; //alpha.y;

      highp vec2 p;
      p.x = pow(r, c)*exp(-d*theta) * cos(d*log(r) + c*theta);
      p.y = pow(r, c)*exp(-d*theta) * sin(d*log(r) + c*theta);

      /*
      if (lookup(p) > 0.0) {
          gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
      } else {
          gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      }
      */

      /*
      for (highp float i = -3.0; i < 4.0; i++) {
          for (highp float j = -3.0; j < 4.0; j++) {
              count += lookup(p + vec2(i*dp - 0.7071067*dp*j, j*dp + 0.7071067*dp*i));
          }
      }

      // highp float c = 1.0 - (count / 49.0);
      highp float c = count / 49.0;
      gl_FragColor = vec4(c, c, c, 1.0);
      */

      highp vec4 c4;
      for (highp float i = -3.0; i < 4.0; i++) {
          for (highp float j = -3.0; j < 4.0; j++) {
              c4 += lookup_droste(p + vec2(i*dp - 0.7071067*dp*j, j*dp + 0.7071067*dp*i));
          }
      }

      gl_FragColor = c4 / 49.0;

      // gl_FragColor = texture2D(uSampler, vTextureCoord);
      // gl_FragColor = lookup_droste(p);
    }
  `;

    // Initialize a shader program; this is where all the lighting
    // for the vertices and so forth is established.
    const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

    // Collect all the info needed to use the shader program.
    // Look up which attributes our shader program is using
    // for aVertexPosition, aTextureCoord and also
    // look up uniform locations.
    const programInfo = {
	program: shaderProgram,
	attribLocations: {
	    vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
	    textureCoord: gl.getAttribLocation(shaderProgram, 'aTextureCoord'),
	},
	uniformLocations: {
	    projectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
	    modelViewMatrix: gl.getUniformLocation(shaderProgram, 'uModelViewMatrix'),
	    uSampler: gl.getUniformLocation(shaderProgram, 'uSampler'),
	    uScale: gl.getUniformLocation(shaderProgram, 'uScale'),
	},
    };

    // Here's where we call the routine that builds all the
    // objects we'll be drawing.
    const buffers = initBuffers(gl);

    const texture = loadTexture(gl, 'base.png');

    var then = 0;

    // Draw the scene repeatedly
    function render(now) {
	now *= 0.001;  // convert to seconds
	const deltaTime = now - then;
	then = now;

	drawScene(gl, programInfo, buffers, texture, deltaTime);

	requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
    // render();
}

//
// initBuffers
//
// Initialize the buffers we'll need. For this demo, we just
// have one object -- a simple three-dimensional cube.
//
function initBuffers(gl) {

    // Create a buffer for the cube's vertex positions.

    const positionBuffer = gl.createBuffer();

    // Select the positionBuffer as the one to apply buffer
    // operations to from here out.

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Now create an array of positions for the cube.

    const positions = [
	// Front face
	-1.0, -1.0,  0.0,
	 1.0, -1.0,  0.0,
	 1.0,  1.0,  0.0,
	-1.0,  1.0,  0.0,
    ];

    // Now pass the list of positions into WebGL to build the
    // shape. We do this by creating a Float32Array from the
    // JavaScript array, then use it to fill the current buffer.

    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // Now set up the texture coordinates for the faces.

    const textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, textureCoordBuffer);

    /*
    const textureCoordinates = [
	// Front
	-1.0, -1.0,
	 1.0, -1.0,
	 1.0,  1.0,
	-1.0,  1.0,
    ];
    */

    const textureCoordinates = [
	// Front
	-1.2, -1.2,
	 1.2, -1.2,
	 1.2,  1.2,
	-1.2,  1.2,
    ];

    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoordinates),
                  gl.STATIC_DRAW);

    // Build the element array buffer; this specifies the indices
    // into the vertex arrays for each face's vertices.

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

    // This array defines each face as two triangles, using the
    // indices into the vertex array to specify each triangle's
    // position.

    const indices = [
	0,  1,  2,      0,  2,  3    // front
    ];

    // Now send the element array to GL

    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
		  new Uint16Array(indices), gl.STATIC_DRAW);

    return {
	position: positionBuffer,
	textureCoord: textureCoordBuffer,
	indices: indexBuffer,
    };
}

//
// Initialize a texture and load an image.
// When the image finished loading copy it into the texture.
//
function loadTexture(gl, url) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Because images have to be download over the internet
    // they might take a moment until they are ready.
    // Until then put a single pixel in the texture so we can
    // use it immediately. When the image has finished downloading
    // we'll update the texture with the contents of the image.
    const level = 0;
    const internalFormat = gl.RGBA;
    const width = 1;
    const height = 1;
    const border = 0;
    const srcFormat = gl.RGBA;
    const srcType = gl.UNSIGNED_BYTE;
    const pixel = new Uint8Array([0, 0, 255, 255]);  // opaque blue
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                  width, height, border, srcFormat, srcType,
                  pixel);

    const image = new Image();
    image.onload = function() {
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                      srcFormat, srcType, image);

	// WebGL1 has different requirements for power of 2 images
	// vs non power of 2 images so check if the image is a
	// power of 2 in both dimensions.
	if (isPowerOf2(image.width) && isPowerOf2(image.height)) {
	    // Yes, it's a power of 2. Generate mips.
	    gl.generateMipmap(gl.TEXTURE_2D);
	} else {
	    // No, it's not a power of 2. Turn of mips and set
	    // wrapping to clamp to edge
	    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	}
    };
    image.src = url;

    return texture;
}

function isPowerOf2(value) {
    return (value & (value - 1)) == 0;
}

//
// Draw the scene.
//
function drawScene(gl, programInfo, buffers, texture, deltaTime) {
    gl.clearColor(0.0, 0.0, 0.0, 1.0);  // Clear to black, fully opaque
    gl.clearDepth(1.0);                 // Clear everything
    gl.enable(gl.DEPTH_TEST);           // Enable depth testing
    gl.depthFunc(gl.LEQUAL);            // Near things obscure far things

    // Clear the canvas before we start drawing on it.

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Create a perspective matrix, a special matrix that is
    // used to simulate the distortion of perspective in a camera.
    // Our field of view is 45 degrees, with a width/height
    // ratio that matches the display size of the canvas
    // and we only want to see objects between 0.1 units
    // and 100 units away from the camera.

    const mat4 = glMatrix.mat4
    const fieldOfView = 45 * Math.PI / 180;   // in radians
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = -1.0;
    const zFar = 1.0;
    const projectionMatrix = mat4.create();

    // note: glmatrix.js always has the first argument
    // as the destination to receive the result.

    //mat4.perspective(projectionMatrix,
    //                 fieldOfView,
    //                 aspect,
    //                 zNear,
    //                 zFar);

    mat4.ortho(projectionMatrix,
               -1.0,
                1.0,
	       -1.0,
	        1.0,
               zNear,
               zFar);

    // Set the drawing position to the "identity" point, which is
    // the center of the scene.
    const modelViewMatrix = mat4.create();

    // Now move the drawing position a bit to where we want to
    // start drawing the square.

    mat4.translate(modelViewMatrix,     // destination matrix
                   modelViewMatrix,     // matrix to translate
                   [0.0, 0.0, 0.0]);  // amount to translate
    mat4.rotate(modelViewMatrix,  // destination matrix
		modelViewMatrix,  // matrix to rotate
		cubeRotation,     // amount to rotate in radians
		[0, 0, 1]);       // axis to rotate around (Z)
    mat4.rotate(modelViewMatrix,  // destination matrix
		modelViewMatrix,  // matrix to rotate
		cubeRotation * .7,// amount to rotate in radians
		[0, 1, 0]);       // axis to rotate around (X)

    // Tell WebGL how to pull out the positions from the position
    // buffer into the vertexPosition attribute
    {
	const numComponents = 3;
	const type = gl.FLOAT;
	const normalize = false;
	const stride = 0;
	const offset = 0;
	gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
	gl.vertexAttribPointer(
            programInfo.attribLocations.vertexPosition,
            numComponents,
            type,
            normalize,
            stride,
            offset);
	gl.enableVertexAttribArray(
            programInfo.attribLocations.vertexPosition);
    }

    // Tell WebGL how to pull out the texture coordinates from
    // the texture coordinate buffer into the textureCoord attribute.
    {
	const numComponents = 2;
	const type = gl.FLOAT;
	const normalize = false;
	const stride = 0;
	const offset = 0;
	gl.bindBuffer(gl.ARRAY_BUFFER, buffers.textureCoord);
	gl.vertexAttribPointer(
            programInfo.attribLocations.textureCoord,
            numComponents,
            type,
            normalize,
            stride,
            offset);
	gl.enableVertexAttribArray(
            programInfo.attribLocations.textureCoord);
    }

    // Tell WebGL which indices to use to index the vertices
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);

    // Tell WebGL to use our program when drawing

    gl.useProgram(programInfo.program);

    // Set the shader uniforms

    gl.uniformMatrix4fv(
	programInfo.uniformLocations.projectionMatrix,
	false,
	projectionMatrix);
    gl.uniformMatrix4fv(
	programInfo.uniformLocations.modelViewMatrix,
	false,
	modelViewMatrix);

    // Specify the texture to map onto the faces.

    // Tell WebGL we want to affect texture unit 0
    gl.activeTexture(gl.TEXTURE0);

    // Bind the texture to texture unit 0
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Tell the shader we bound the texture to texture unit 0
    gl.uniform1i(programInfo.uniformLocations.uSampler, 0);

    gl.uniform1f(programInfo.uniformLocations.uScale, Math.exp(-gridScale));

    {
	const vertexCount = 6;
	const type = gl.UNSIGNED_SHORT;
	const offset = 0;
	gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
    }

    // Update the rotation for the next draw

    gridScale += 0.1 * deltaTime;

    if (gridScale > 5) {
	gridscale = 0;
    }
}

//
// Initialize a shader program, so WebGL knows how to draw our data
//
function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

    // Create the shader program

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    // If creating the shader program failed, alert

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
	alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
	return null;
    }

    return shaderProgram;
}

//
// creates a shader of the given type, uploads the source and
// compiles it.
//
function loadShader(gl, type, source) {
    const shader = gl.createShader(type);

    // Send the source to the shader object

    gl.shaderSource(shader, source);

    // Compile the shader program

    gl.compileShader(shader);

    // See if it compiled successfully

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
	alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
	gl.deleteShader(shader);
	return null;
    }

    return shader;
}
