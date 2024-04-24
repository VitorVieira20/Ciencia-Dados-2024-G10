function getRoomsValue() {
  var uiRooms = document.getElementsByName("uiRooms");
  for(var i in uiRooms) {
    if(uiRooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1;
}

function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for(var i in uiBathrooms) {
    if(uiBathrooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1;
}

function getBalconyValue() {
  var uiBalcony = document.getElementsByName("uiBalcony");
  for(var i in uiBalcony) {
    if(uiBalcony[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1;
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft").value;
    var rooms = getRoomsValue();
    var bath = getBathValue();
    var balcony = getBalconyValue();
    var location = document.getElementsByClassName("location")[0].value; // Alteração aqui
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5000/predict_house_price";

    $.post(url, {
        location: location,
        total_sqft: parseFloat(sqft),
        rooms: rooms,
        bath: bath,
        balcony: balcony
    },function(data, status) {
      console.log(data.predicted_price); // Alterado de estimated_price para predicted_price
      estPrice.innerHTML = "<h2>" + data.predicted_price.toString() + " Lakh</h2>"; // Alterado de estimated_price para predicted_price
      console.log(status);
    });
}

function onPageLoad() {
    var url = "http://127.0.0.1:5000/get_location_names";
    $.get(url,function(data, status) {
        console.log("got response for get_location_names request");
        if(data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for(var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
}

window.onload = onPageLoad;