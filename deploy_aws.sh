#!/bin/bash
# ==============================================================================
# Script de Despliegue Automático en AWS EC2 (Docker + FastAPI + MLOps)
# Optimizado para AWS Free Tier (Capa Gratuita) en cualquier Región
# Laboratorios PiSA S.A. de C.V. / SalmiBC
# ==============================================================================
set -e

echo "=================================================================="
echo "🚀 INICIANDO DESPLIEGUE AUTOMÁTICO EN AWS (FREE TIER)"
echo "=================================================================="

# 1. Detectar Región activa (CloudShell expone $AWS_REGION)
REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || echo "us-east-2")}}
if [ -z "$REGION" ] || [ "$REGION" == "None" ]; then REGION="us-east-2"; fi
echo "[1/6] Región activa detectada: $REGION"

# 2. Obtener VPC y Subnet por defecto
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text 2>/dev/null || true)
if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    VPC_ID=$(aws ec2 describe-vpcs --region $REGION --query "Vpcs[0].VpcId" --output text 2>/dev/null | awk '{print $1}')
fi

SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text 2>/dev/null | awk '{print $1}')
echo "[2/6] Red VPC: $VPC_ID | Subnet: $SUBNET_ID"

# 3. Crear o reutilizar Grupo de Seguridad (abrir puertos 80 y 8000)
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

# 4. Buscar AMI oficial de Amazon Linux 2023 para la región
AMI_ID=$(aws ec2 describe-images --region $REGION --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
echo "[4/6] Imagen AMI oficial: $AMI_ID"

# 5. Detectar tipo de instancia Free Tier para la cuenta (t3.micro o t2.micro)
FREE_TIER_TYPE=$(aws ec2 describe-instance-types --region $REGION --filters "Name=free-tier-eligible,Values=true" --query "InstanceTypes[?CurrentGeneration==\`true\`].InstanceType | [0]" --output text 2>/dev/null || true)
if [ "$FREE_TIER_TYPE" == "None" ] || [ -z "$FREE_TIER_TYPE" ]; then
    # Fallback automático: en us-east-2 t3.micro es el Free Tier nativo
    FREE_TIER_TYPE="t3.micro"
fi
echo "[5/6] Tipo de instancia Free Tier oficial: $FREE_TIER_TYPE"

# 6. Preparar User Data con Swap para asegurar instalación de Docker sin saturar RAM
cat << 'USERDATA_EOF' > /tmp/oee_userdata.sh
#!/bin/bash
# Habilitar 2GB de Swap para garantizar memoria durante la instalación de paquetes
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile swap swap defaults 0 0" >> /etc/fstab

# Instalar Docker y Git
dnf update -y
dnf install -y docker git
systemctl start docker
systemctl enable docker

# Crear directorio de trabajo
mkdir -p /opt/oee-app
cd /opt/oee-app

# Clonar proyecto de GitHub
git clone https://github.com/SalmiBC/26_proyecto_OEE.git .

# Compilar contenedor Docker optimizado
docker build -t oee-api .

# Arrancar contenedor mapeando puerto 80 (Web directa) y 8000 (API)
docker run -d --restart always -p 80:8000 -p 8000:8000 --name oee-service oee-api
USERDATA_EOF

# 7. Lanzar la instancia en AWS
echo "Lanzando instancia $FREE_TIER_TYPE en $REGION..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $FREE_TIER_TYPE \
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
echo "Región       : $REGION"
echo "Instancia ID : $INSTANCE_ID"
echo "Tipo         : $FREE_TIER_TYPE (Free Tier $0 USD)"
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
