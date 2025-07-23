var QWebChannelMessageTypes = {
  signal: 1,
  propertyUpdate: 2,
  init: 3,
  idle: 4,
  debug: 5,
  invokeMethod: 6,
  connectToSignal: 7,
  disconnectFromSignal: 8,
  setProperty: 9,
  response: 10,
}

var QWebChannel = function (transport, initCallback) {
  if (typeof transport !== "object" || typeof transport.send !== "function") {
    console.error(
      "The QWebChannel expects a transport object with a send function and onmessage callback property." +
        " Given is: transport: " +
        typeof transport +
        ", transport.send: " +
        typeof transport.send,
    )
    return
  }
  
  this.transport = transport

  this.send = (data) => {
    if (typeof data !== "string") {
      data = JSON.stringify(data)
    }
    this.transport.send(data)
  }

  this.transport.onmessage = (message) => {
    var data = message.data
    if (typeof data === "string") {
      data = JSON.parse(data)
    }
    switch (data.type) {
      case QWebChannelMessageTypes.signal:
        this.handleSignal(data)
        break
      case QWebChannelMessageTypes.response:
        this.handleResponse(data)
        break
      case QWebChannelMessageTypes.propertyUpdate:
        this.handlePropertyUpdate(data)
        break
      default:
        console.error("invalid message received:", message.data)
        break
    }
  }

  this.execCallbacks = {}
  this.execId = 0
  this.exec = (data, callback) => {
    if (!callback) {
      // if no callback is given, send directly
      this.send(data)
      return
    }
    if (this.execId === Number.MAX_VALUE) {
      // wrap
      this.execId = Number.MIN_VALUE
    }
    if (data.hasOwnProperty("id")) {
      console.error("Cannot exec message with property id: " + JSON.stringify(data))
      return
    }
    data.id = this.execId++
    this.execCallbacks[data.id] = callback
    this.send(data)
  }

  this.objects = {}

  this.handleSignal = (message) => {
    var object = this.objects[message.object]
    if (object) {
      object.signalEmitted(message.signal, message.args)
    } else {
      console.warn("Unhandled signal: " + message.object + "::" + message.signal)
    }
  }

  this.handleResponse = (message) => {
    if (!message.hasOwnProperty("id")) {
      console.error("Invalid response message received: ", JSON.stringify(message))
      return
    }
    this.execCallbacks[message.id](message.data)
    delete this.execCallbacks[message.id]
  }

  this.handlePropertyUpdate = (message) => {
    for (var i in message.data) {
      var data = message.data[i]
      var object = this.objects[data.object]
      if (object) {
        object.propertyUpdate(data.signals, data.properties)
      } else {
        console.warn("Unhandled property update: " + data.object)
      }
    }
    this.exec({ type: QWebChannelMessageTypes.idle })
  }

  this.debug = (message) => {
    this.send({ type: QWebChannelMessageTypes.debug, data: message })
  }

  var initCallbackWrapper = (data) => {
    for (var objectName in data) {
      var obj = new QObject(objectName, data[objectName], this)
    }
    // now unwrap properties, which might reference other registered objects
    for (var objName in this.objects) {
      this.objects[objName].unwrapProperties()
    }
    if (initCallback) {
      initCallback(this)
    }
    this.exec({ type: QWebChannelMessageTypes.idle })
  }

  this.exec({ type: QWebChannelMessageTypes.init }, initCallbackWrapper)
}

function QObject(name, data, webChannel) {
  this.__id__ = name
  webChannel.objects[name] = this

  // List of callbacks that get invoked upon signal emission
  this.__objectSignals__ = {}

  // Cache of all properties, updated when a notify signal is emitted
  this.__propertyCache__ = {}

  var object = this

  // ----------------------------------------------------------------------

  this.unwrapQObject = (response) => {
    if (response instanceof Array) {
      // support list of objects
      var ret = new Array(response.length)
      for (var i = 0; i < response.length; ++i) {
        ret[i] = object.unwrapQObject(response[i])
      }
      return ret
    }
    if (!response || !response["__QObject*__"] || response.id === undefined) {
      return response
    }

    var objectId = response.id
    if (webChannel.objects[objectId]) return webChannel.objects[objectId]

    if (!response.data) {
      console.error("Cannot unwrap unknown QObject " + objectId + " without data.")
      return
    }

    var qObject = new QObject(objectId, response.data, webChannel)
    qObject.destroyed.connect(() => {
      if (webChannel.objects[objectId] === qObject) {
        delete webChannel.objects[objectId]
        // reset the now deleted QObject to an empty {} object
        // just assigning {} though would not have the desired effect, but the
        // below also ensures all external references will see the empty map
        // NOTE: this detour is necessary to workaround QTBUG-40021
        var propertyNames = []
        for (var propertyName in qObject) {
          propertyNames.push(propertyName)
        }
        for (var idx in propertyNames) {
          delete qObject[propertyNames[idx]]
        }
      }
    })
    // here we are already initialized, and thus must directly unwrap the properties
    qObject.unwrapProperties()
    return qObject
  }

  this.unwrapProperties = () => {
    for (var propertyIdx in object.__propertyCache__) {
      object.__propertyCache__[propertyIdx] = object.unwrapQObject(object.__propertyCache__[propertyIdx])
    }
  }

  function addSignal(signalData, isPropertyNotifySignal) {
    var signalName = signalData[0]
    var signalIndex = signalData[1]
    object[signalName] = {
      connect: (callback) => {
        if (typeof callback !== "function") {
          console.error("Bad callback given to connect to signal " + signalName)
          return
        }

        object.__objectSignals__[signalIndex] = object.__objectSignals__[signalIndex] || []
        object.__objectSignals__[signalIndex].push(callback)

        if (!isPropertyNotifySignal && signalName !== "destroyed") {
          // only required for "pure" signals, handled separately for properties in propertyUpdate
          // also note that we always get notified about the destroyed signal
          webChannel.exec({
            type: QWebChannelMessageTypes.connectToSignal,
            object: object.__id__,
            signal: signalIndex,
          })
        }
      },
      disconnect: (callback) => {
        if (typeof callback !== "function") {
          console.error("Bad callback given to disconnect from signal " + signalName)
          return
        }
        object.__objectSignals__[signalIndex] = object.__objectSignals__[signalIndex] || []
        var idx = object.__objectSignals__[signalIndex].indexOf(callback)
        if (idx === -1) {
          console.error("Cannot find connection of signal " + signalName + " to " + callback.name)
          return
        }
        object.__objectSignals__[signalIndex].splice(idx, 1)
        if (!isPropertyNotifySignal && object.__objectSignals__[signalIndex].length === 0) {
          // only required for "pure" signals, handled separately for properties in propertyUpdate
          webChannel.exec({
            type: QWebChannelMessageTypes.disconnectFromSignal,
            object: object.__id__,
            signal: signalIndex,
          })
        }
      },
    }
  }

  /**
   * Invokes all callbacks for the given signalname. Also works for property notify callbacks.
   */
  function invokeSignalCallbacks(signalName, signalArgs) {
    var connections = object.__objectSignals__[signalName]
    if (connections) {
      connections.forEach((callback) => {
        callback.apply(callback, signalArgs)
      })
    }
  }

  this.propertyUpdate = (signals, propertyMap) => {
    // update property cache
    for (var propertyIndex in propertyMap) {
      var propertyValue = propertyMap[propertyIndex]
      object.__propertyCache__[propertyIndex] = propertyValue
    }

    for (var signalName in signals) {
      // Invoke all callbacks, as signalEmitted() does not. This ensures the
      // property cache is updated before the callbacks are invoked.
      invokeSignalCallbacks(signalName, signals[signalName])
    }
  }

  this.signalEmitted = (signalName, signalArgs) => {
    invokeSignalCallbacks(signalName, signalArgs)
  }

  function addMethod(methodData) {
    var methodName = methodData[0]
    var methodIdx = methodData[1]
    object[methodName] = () => {
      var args = []
      var callback
      for (var i = 0; i < arguments.length; ++i) {
        if (typeof arguments[i] === "function") callback = arguments[i]
        else args.push(arguments[i])
      }

      webChannel.exec(
        {
          type: QWebChannelMessageTypes.invokeMethod,
          object: object.__id__,
          method: methodIdx,
          args: args,
        },
        (response) => {
          if (response !== undefined) {
            var result = object.unwrapQObject(response)
            if (callback) {
              callback(result)
            }
          }
        },
      )
    }
  }

  function bindGetterSetter(propertyInfo) {
    var propertyIndex = propertyInfo[0]
    var propertyName = propertyInfo[1]
    var notifySignalIndex = propertyInfo[2]
    // initialize property cache with current value
    // NOTE: if this is an object, it is not directly unwrapped as it might
    // reference other QObject that we do not know yet
    object.__propertyCache__[propertyIndex] = propertyInfo[3]

    if (notifySignalIndex) {
      var notifySignal = {
        index: notifySignalIndex,
        name: "notify::" + propertyName,
      }
      addSignal(notifySignal, true)
    }

    Object.defineProperty(object, propertyName, {
      configurable: true,
      get: () => {
        var propertyValue = object.__propertyCache__[propertyIndex]
        if (propertyValue === undefined) {
          // This shouldn't happen
          console.warn(
            'Undefined value in property cache for property "' + propertyName + '" in object ' + object.__id__,
          )
        }

        return propertyValue
      },
      set: (value) => {
        if (propertyValue === value) {
          return
        }
        var propertyValue = object.__propertyCache__[propertyIndex]
        if (propertyValue === value) {
          return
        }

        object.__propertyCache__[propertyIndex] = value
        webChannel.exec({
          type: QWebChannelMessageTypes.setProperty,
          object: object.__id__,
          property: propertyIndex,
          value: value,
        })
      },
    })
  }

  // ----------------------------------------------------------------------

  data.methods.forEach(addMethod)

  data.properties.forEach(bindGetterSetter)

  data.signals.forEach((signal) => {
    addSignal(signal, false)
  })

  for (var enumName in data.enums) {
    object[enumName] = data.enums[enumName]
  }
}

//required for use with nodejs
if (typeof module === "object") {
  module.exports = {
    QWebChannel: QWebChannel,
  }
}
