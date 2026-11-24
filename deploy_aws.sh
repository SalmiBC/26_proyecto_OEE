#!/bin/bash
# ==============================================================================
# Script de Despliegue Automático en AWS EC2 (Docker + FastAPI + MLOps)
# Laboratorios PiSA S.A. de C.V. / SalmiBC
# ==============================================================================
set -e

echo "=========================================================="
echo "🚀 INICIANDO DESPLIEGUE AUTOMÁTICO EN AWS"
echo "=========================================================="

# 1. Detectar Región y VPC por defecto
REGION=$(aws configure get region || echo "us-east-1")
if [ -z "$REGION" ]; then REGION="us-east-1"; fi
echo "[1/6] Región activa: $REGION"

VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    VPC_ID=$(aws ec2 describe-vpcs --region $REGION --query "Vpcs[0].VpcId" --output text)
fi
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text)
echo "[2/6] VPC: $VPC_ID | Subnet: $SUBNET_ID"

# 2. Crear o reutilizar Grupo de Seguridad (abrir puertos 80 y 8000)
SG_NAME="oee-predictivo-sg"
SG_ID=$(aws ec2 describe-security-groups --region $REGION --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "[3/6] Creando Grupo de Seguridad: $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "Security Group para OEE FastAPI y Docker" --vpc-id $VPC_ID --query "GroupId" --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
else
    echo "[3/6] Usando Grupo de Seguridad existente: $SG_ID"
fi

# 3. Buscar la imagen oficial más reciente de Amazon Linux 2023
AMI_ID=$(aws ec2 describe-images --region $REGION --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
echo "[4/6] AMI oficial de Amazon Linux: $AMI_ID"

# 4. Crear script de arranque automatizado de la máquina (User Data)
cat << 'USERDATA_EOF' > /tmp/oee_userdata.sh
#!/bin/bash
dnf update -y
dnf install -y docker git
systemctl start docker
systemctl enable docker

# Crear directorio de la app
mkdir -p /opt/oee-app
cd /opt/oee-app

# Clonar repositorio público de GitHub
git clone https://github.com/SalmiBC/26_proyecto_OEE.git .

# Construir contenedor Docker optimizado
docker build -t oee-api .

# Ejecutar el contenedor mapeando puerto 80 (HTTP estándar) y 8000
docker run -d --restart always -p 80:8000 -p 8000:8000 --name oee-service oee-api
USERDATA_EOF

# 5. Lanzar la instancia EC2 t2.micro (Elegible para la capa gratuita de AWS)
echo "[5/6] Lanzando instancia EC2 t2.micro (Free Tier)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type t2.micro \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --user-data file:///tmp/oee_userdata.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=oee-predictivo-pisa}]' \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Instancia creada con ID: $INSTANCE_ID"
echo "[6/6] Esperando a que AWS inicialice la instancia y asigne la IP pública..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
PUBLIC_DNS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicDnsName" --output text)

echo ""
echo "=================================================================="
echo "🎉 ¡DESPLIEGUE EN AWS COMPLETADO EXITOSAMENTE!"
echo "=================================================================="
echo "Instancia ID : $INSTANCE_ID"
echo "IP Pública   : $PUBLIC_IP"
echo "DNS Público  : $PUBLIC_DNS"
echo ""
echo "En aproximadamente 2 minutos (mientras Docker termina de compilar),"
echo "tu servicio estará disponible en:"
echo ""
echo "🌐 Aplicación Web     : http://$PUBLIC_IP"
echo "📚 Swagger OpenAPI   : http://$PUBLIC_IP/docs"
echo "🩺 Health Check      : http://$PUBLIC_IP/health"
echo "📊 Presentación Rúbrica: http://$PUBLIC_IP/presentacion"
echo "=================================================================="
