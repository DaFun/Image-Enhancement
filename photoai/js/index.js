var dataURL;
var server = "http://184.105.86.228:9999/infer";

function readURL(input) {
  $('#res img:last-child').remove();
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();

      $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
      dataURL = reader.result;
      $('#mode').show();
    };

    reader.readAsDataURL(input.files[0]);
    
  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
  $('#mode').hide();
  $('#res img:last-child').remove();
}


$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});


function serverRequest(mode) {
  $.ajax({
    type: 'POST',
    url: server,
    crossDomain: true,
    data: {
        mode: mode,
        data: dataURL
    },
    dataType: 'json',
    success: function(data) {
  	 $("<img>", {
         "src": "data:image/jpeg;base64," + data['data'],
         "class": "file-upload-image"
         }).appendTo("#res")
//     $('#res').attr('src', "data:image/jpeg;base64," + data['data'])
  	},
  	error: function(jqXHR, textStatus, errorThrown){
       console.log('error')
       console.log(jqXHR)
       console.log(textStatus)
       console.log(errorThrown)
    }
  })
}