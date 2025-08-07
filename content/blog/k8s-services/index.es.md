---
title: "Tipos de Servicios en Kubernetes"
# series: ["Kubernetes"]
# series_order: 1
authors:
  - jnonino
description: >
  En este artículo vamos a repasar los cuatro tipos de servicios que están disponibles en Kubernetes, ClusterIP, NodePort, LoadBalancer y ExternalName. Revisaremos sus detalles y cuando se debe utilizar cada uno.
date: 2025-08-05
tags: ["Blog","Kubernetes", "Services"]
---

{{< lead >}}
Luego de desplegar una aplicación (pod/s) en Kubernetes, independientemente del método elegido para ello, si queremos que sea alcanzable por otras aplicaciones necesitamos crear un servicio. En este artículo vamos a repasar los cuatro tipos de servicios que están disponibles en Kubernetes, ClusterIP, NodePort, LoadBalancer y ExternalName. Revisaremos sus detalles y cuando se debe utilizar cada uno.
{{< /lead >}}

El tipo de servicio se especifica mediante el valor `spec.type`[^1]:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: servicio-de-prueba
spec:
  type: <TIPO_DE_SERVICIO>
```

A continuación veremos cada uno de ellos.

## `type: ClusterIP`

Es el tipo de servicio por defecto. Kubernetes asigna una dirección IP al servicio que las demás aplicaciones pueden utilizar para comunicarse. La dirección asignada proviene de un grupo de direcciones reservadas para este fin mediante el valor `service-cluster-ip-range` en el servidor de API de Kubernetes.

Cuando se crea un servicio del tipo `ClusterIP` el servicio no es accesible desde el exterior y los pods detrás de este servicio solo pueden ser contactados por otros pods del mismo cluster.

## `type: NodePort`



## `type: LoadBalancer`



## `type: ExternalName`


---

## Referencias

[^1]: [Documentación oficial de Kubernetes](https://kubernetes.io/docs/concepts/services-networking/service/)
